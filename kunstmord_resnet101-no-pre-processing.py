import pandas as pd

import cv2

import time

import torchvision

import torch.nn as nn

from tqdm import tqdm_notebook as tqdm

import scipy as sp



from sklearn import metrics

from functools import partial



from PIL import Image, ImageFile

from torch.utils.data import Dataset

import torch

import torch.optim as optim

from torchvision import transforms

from torch.optim import lr_scheduler

import os

import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold

from scipy import stats

import random



import albumentations

from albumentations import torch as AT
device = torch.device("cuda:0")

ImageFile.LOAD_TRUNCATED_IMAGES = True
class RetinopathyDataset_v2(Dataset):



    def __init__(self, df, transform, train=True):



        self.data = df.reset_index()

        

        if train:

            self.prefix = "train"

        else:

            self.prefix = "test"

            

        self.transform = transform



    def __len__(self):

        return len(self.data)

    



    def __getitem__(self, idx):

        img_name = os.path.join('../input/aptos2019-blindness-detection/{}_images'.format(self.prefix),

                                self.data.loc[idx, 'id_code'] + '.png')

        

        img = cv2.imread(img_name)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image = self.transform(image=img)

        image = image['image']

        

        label = torch.tensor(self.data.loc[idx, 'diagnosis'])

        

        return {'image': image,

                'labels': label

                }
class RetinopathyDataset(Dataset):



    def __init__(self, csv_file, transform, train=True):



        self.data = pd.read_csv(csv_file)

        

        if train:

            self.prefix = "train"

        else:

            self.prefix = "test"

            

        self.transform = transform



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        img_name = os.path.join('../input/aptos2019-blindness-detection/{}_images'.format(self.prefix), self.data.loc[idx, 'id_code'] + '.png')

       

        img = cv2.imread(img_name)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image = self.transform(image=img)

        image = image['image']

        

        label = torch.tensor(self.data.loc[idx, 'diagnosis'])

        return {'image': image,

                'labels': label

                }
class OptimizedRounder(object):

    def __init__(self):

        self.coef_ = 0



    def _kappa_loss(self, coef, X, y):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4



        ll = metrics.cohen_kappa_score(y, X_p, weights='quadratic')

        return -ll



    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X=X, y=y)

        initial_coef = [0.5, 1.5, 2.5, 3.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')



    def predict(self, X, coef):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4

        return X_p



    def coefficients(self):

        return self.coef_['x']
model = torchvision.models.resnet101(pretrained=False)

model.load_state_dict(torch.load("../input/fastai-pretrained-models/resnet101-5d3b4d8f.pth"))



for param in model.parameters():

    param.requires_grad = False



model.fc =  nn.Sequential(

                          nn.Linear(in_features=2048, out_features=1, bias=True),

                         )



model = model.to(device)
transforms_train = albumentations.Compose([

    albumentations.Resize(224, 224),

    albumentations.HorizontalFlip(),

    albumentations.VerticalFlip(),

    albumentations.RandomBrightness(),

    albumentations.ShiftScaleRotate(rotate_limit=20, scale_limit=0.10),

    albumentations.Normalize(),

    AT.ToTensor()

])



transforms_validation = albumentations.Compose([

    albumentations.Resize(224, 224),

    albumentations.Normalize(),

    AT.ToTensor()

])





transforms_test = albumentations.Compose([

    albumentations.Resize(224, 224),

    albumentations.Rotate(limit=10),

    albumentations.Normalize(),

    AT.ToTensor()

])
train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')



skf = StratifiedKFold(n_splits=10, random_state=2052)



for i, (train_index, val_index) in enumerate(skf.split(train_df['diagnosis'], train_df['diagnosis'])):

    # we want to have a 90/10 split, so we use a Stratified 10-fold split and just take the first split

    if i == 0:

        validation_df = train_df.iloc[val_index]

        train_df = train_df.iloc[train_index]



train_dataset = RetinopathyDataset_v2(df=train_df, train=True, transform=transforms_train)

validation_dataset = RetinopathyDataset_v2(df=validation_df, train=True, transform=transforms_validation)

test_dataset = RetinopathyDataset(csv_file='../input/aptos2019-blindness-detection/sample_submission.csv',

                                  transform=transforms_test, train=False)
# we have a class imbalance, so we will weight the classes differently, this computes a weighted MSE loss



def weighted_mse_loss(preds, target, weight):

    return torch.sum(weight * (preds - target) ** 2)
data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

data_loader_val = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=True, num_workers=4)



num_epochs = 25



optimizer = optim.Adam(model.parameters(), lr=0.0002)



# going to be doing MixUp, this controls the beta distribution of the mixing ratio

alpha_mixup = 0.3



# train top FC layer for n_freeze epochs, then un-freeze all other layers and train the full Resnet

n_freeze = 2



# we want class weights to be inversely proportional to the frequency of the classes

# weight(class) ~ (proportion of class in dataset)^(-weight_pow)

weight_pow = 0.3

label_counts = train_df['diagnosis'].value_counts().values

label_weights = (label_counts[:] / label_counts.sum())**(-weight_pow)



# normalize so that 1st class has a weight of 1

label_weights /= label_weights[0]

print(label_weights)
since = time.time()



criterion = weighted_mse_loss



best = 0

nep = 0

best_coeff = []



month_day = time.gmtime()[1:3]



for epoch in range(num_epochs):

    print('Epoch {}/{}'.format(epoch, num_epochs - 1))

    print('-' * 10)

    model.train()

    running_loss = 0.0

    tk0 = tqdm(data_loader, total=int(len(data_loader)))

    counter = 0

    

    if epoch == n_freeze:                

        for param in model.parameters():

            param.requires_grad = True 

    

    for bi, d in enumerate(tk0):

        inputs = d["image"]

        labels = d["labels"].view(-1, 1)

        weights = labels[:, 0].clone()

                

        for i in range(len(label_weights)):

            weights[labels[:, 0] == i] = label_weights[i]



        # we won't load a different batch for the mixup, just shuffle the current one

        # and mix it with the unshuffled version

        shuffled_index = list(range(inputs.shape[0]))

        random.shuffle(shuffled_index)



        mixed_up_inputs = inputs[shuffled_index, :, :, :]

        mixed_up_labels = labels[shuffled_index, :]



        mixed_up_weights = mixed_up_labels[:, 0].clone()



        for i in range(len(label_weights)):

            mixed_up_weights[mixed_up_labels[:, 0] == i] = label_weights[i]



        l_mixup = stats.beta.rvs(a=alpha_mixup, b=alpha_mixup, size=inputs.shape[0])



        inputs = torch.as_tensor(l_mixup, dtype=torch.float).view(-1, 1, 1, 1) * inputs

        inputs += torch.as_tensor(1. - l_mixup, dtype=torch.float).view(-1, 1, 1, 1) * mixed_up_inputs



        labels = labels.float()

        mixed_up_labels = mixed_up_labels.float()



        labels = torch.as_tensor(l_mixup, dtype=torch.float).view(-1, 1) * labels

        labels += torch.as_tensor(1. - l_mixup, dtype=torch.float).view(-1, 1) * mixed_up_labels



        weights = weights.float()

        mixed_up_weights = mixed_up_weights.float()

        weights = torch.as_tensor(l_mixup, dtype=torch.float).view(-1, 1) * weights

        weights += torch.as_tensor(1. - l_mixup, dtype=torch.float).view(-1, 1) * mixed_up_weights



        inputs = inputs.to(device, dtype=torch.float)

        labels = labels.to(device, dtype=torch.float)

        weights = weights.to(device, dtype=torch.float)

        optimizer.zero_grad()

        

        with torch.set_grad_enabled(True):

            outputs = model(inputs)

                

            loss = criterion(outputs, labels, weights)

            loss.backward()

            optimizer.step()

            

        running_loss += loss.item() * inputs.size(0)

        counter += 1

        tk0.set_postfix(loss=(running_loss / (counter * data_loader.batch_size)))

    

    epoch_loss = running_loss / len(data_loader)

    print('Training Loss: {:.4f}'.format(epoch_loss))

    

    model.eval()

    counter = 0

    val_loss = 0.0

    

    preds = []

    full_labels = []

    

    for d in data_loader_val:

        inputs = d["image"]

        labels = d["labels"].view(-1, 1)

        inputs = inputs.to(device, dtype=torch.float)

        with torch.set_grad_enabled(False):

            outputs = model(inputs)

            preds.append(outputs.cpu().numpy())

            full_labels.append(labels.numpy())

            

    preds = np.vstack(preds)[:, 0]

    full_labels = np.vstack(full_labels)[:, 0]

    

    optR = OptimizedRounder()

    optR.fit(preds, full_labels)

    coefficients = optR.coefficients()

    valid_predictions = optR.predict(preds, coefficients)

    

    val_loss = metrics.cohen_kappa_score(valid_predictions, full_labels, weights='quadratic')

    print('Validation Loss: {:.4f}'.format(val_loss))

    if val_loss > best:

        best = val_loss

        nep = epoch

        best_coeff = coefficients[:]

        print('Current best', best, nep, best_coeff)

        torch.save(model.state_dict(), "model_best.pth")



    

time_elapsed = time.time() - since

print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
# sometimes the last threshold would very close to the next-to-last threshold, so I manually shift it a bit

if best_coeff[-1] < 3.0:

    best_coeff[-1] = 3.0
del inputs, labels, weights, mixed_up_inputs, mixed_up_labels, mixed_up_weights, outputs

del train_dataset, train_df, validation_dataset, validation_df

torch.cuda.empty_cache()
model.load_state_dict(torch.load("model_best.pth"))

model.eval()
# number of TTA runs we'll be doing

n_tta = 10



# sometimes got submission errors when running with larger batch size, so probably OOM issue

bsize = 4
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bsize, shuffle=False, num_workers=4)

test_preds = np.zeros((len(test_dataset), 1))



for j in range(n_tta):

    tk0 = tqdm(test_data_loader)

    for i, x_batch in enumerate(tk0):

        x_batch = x_batch["image"]

        pred = model(x_batch.to(device))

        test_preds[i * bsize:(i + 1) * bsize] += pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)

        

test_preds /= n_tta
for i, pred in enumerate(test_preds):

    if pred < best_coeff[0]:

        test_preds[i] = 0

    elif pred >= best_coeff[0] and pred < best_coeff[1]:

        test_preds[i] = 1

    elif pred >= best_coeff[1] and pred < best_coeff[2]:

        test_preds[i] = 2

    elif pred >= best_coeff[2] and pred < best_coeff[3]:

        test_preds[i] = 3

    else:

        test_preds[i] = 4

        

sample = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")

sample.diagnosis = test_preds.astype(int)

sample.to_csv("submission.csv", index=False)