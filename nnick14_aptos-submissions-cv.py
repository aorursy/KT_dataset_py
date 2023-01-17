# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        break

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import sys

package_path = '../input/efficientnet/efficientnet-pytorch/EfficientNet-PyTorch/'

sys.path.append(package_path)
import torch

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

import torch.nn.functional as F

import numpy as np

import torchvision

from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt

import time

import os

import copy

from sklearn.metrics import confusion_matrix

from torch.utils.data import random_split, Dataset

from torch.utils.data.sampler import SubsetRandomSampler

import os

from PIL import Image

import glob

import cv2

import gc #garbage collector for gpu memory 

from tqdm import tqdm

import seaborn as sns

from sklearn.metrics import confusion_matrix, cohen_kappa_score

import pandas as pd

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm



from efficientnet_pytorch import EfficientNet
transform = transforms.Compose(

        [transforms.Resize((320,320)),

        #transforms.RandomCrop(),

        #transforms.RandomHorizontalFlip(.33),

        #transforms.RandomVerticalFlip(.33),

        transforms.RandomApply([

        #transforms.ColorJitter(brightness=(1,3), contrast=(1,3), saturation=(1,2)),

        transforms.RandomAffine(degrees = (-360,360), shear = (-45,45))],

        p=.5),

        transforms.ToTensor(),

        transforms.Normalize([0.460, 0.247, 0.080], [0.249, 0.138, 0.081])

        ])





train_transform = transforms.Compose(

        [transforms.Resize((320,320)),

        #transforms.RandomCrop(),

        #transforms.RandomHorizontalFlip(.33),

        #transforms.RandomVerticalFlip(.33),

        transforms.RandomApply([

        #transforms.ColorJitter(brightness=(1,3), contrast=(1,3), saturation=(1,2)),

        transforms.RandomAffine(degrees = (-360,360), shear = (-45,45))],

        p=.5),

        transforms.ToTensor(),

        transforms.Normalize([0.460, 0.247, 0.080], [0.249, 0.138, 0.081])

        ])

valid_transform = transforms.Compose(

        [transforms.Resize((320,320)),

        #transforms.RandomCrop(),

        #transforms.RandomHorizontalFlip(.33),

        #transforms.RandomVerticalFlip(.33),

        #transforms.ColorJitter(brightness=(1,3), contrast=(1,3), saturation=(1,2)),

        transforms.ToTensor(),

        transforms.Normalize([0.460, 0.247, 0.080], [0.249, 0.138, 0.081])

        ])     



test_transform = transforms.Compose(

        [transforms.Resize((320,320)),

        #transforms.RandomCrop(),

        #transforms.RandomHorizontalFlip(.33),

        #transforms.RandomVerticalFlip(.33),

        #transforms.ColorJitter(brightness=(1,3), contrast=(1,3), saturation=(1,2)),

        transforms.ToTensor(),

        transforms.Normalize([0.460, 0.247, 0.080], [0.249, 0.138, 0.081])

        ])  
class APTOSDataset(Dataset):

    """Eye images dataset."""

    def __init__(self, csv_file, filetype, transform=None):

        self.eye_frame = pd.read_csv(csv_file)

        self.filetype = filetype

        self.transform = transform



    def __len__(self):

        return len(self.eye_frame)

    

    def __getitem__(self, idx):

        if self.filetype == 'train':

            img_name = os.path.join('../input/aptos2019-blindness-detection/train_images',

                                    self.eye_frame.loc[idx,'id_code'] + '.png')



            image = Image.open(img_name)

            if self.transform:

                image = self.transform(image)

            else:

                image = transforms.ToTensor()(image)



            return image,self.eye_frame.diagnosis[idx]

        

        else:

            img_name = os.path.join('../input/aptos2019-blindness-detection/test_images',

                                    self.eye_frame.loc[idx,'id_code'] + '.png')

            image = Image.open(img_name)

            if self.transform:

                image = self.transform(image)

            else:

                image = transforms.ToTensor()(image)

            return image, self.eye_frame.loc[idx,'id_code']

    
class KFoldDataset(Dataset):

    """Eye images dataset."""

    def __init__(self, csv_file, indices, filetype, transform=None):

        self.eye_frame = pd.read_csv(csv_file).iloc[indices].reset_index(drop=True)

        self.filetype = filetype

        self.transform = transform



    def __len__(self):

        return len(self.eye_frame)

    

    def __getitem__(self, idx):

        if self.filetype == 'train':

            img_name = os.path.join('../input/aptos2019-blindness-detection/train_images',

                                    self.eye_frame.loc[idx,'id_code'] + '.png')



            image = Image.open(img_name)

            if self.transform:

                image = self.transform(image)

            else:

                image = transforms.ToTensor()(image)



            return image,self.eye_frame.diagnosis[idx]

        

        else:

            img_name = os.path.join('../input/aptos2019-blindness-detection/test_images',

                                    self.eye_frame.loc[idx,'id_code'] + '.png')

            image = Image.open(img_name)

            if self.transform:

                image = self.transform(image)

            else:

                image = transforms.ToTensor()(image)

            return image, self.eye_frame.loc[idx,'id_code']
"""

train_dataset = APTOSDataset(csv_file='../input/aptos2019-blindness-detection/train.csv',filetype='train',transform=train_transform)

test_dataset = APTOSDataset(csv_file='../input/aptos2019-blindness-detection/test.csv', filetype='test',transform=test_transform)



train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)"""
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_name('efficientnet-b2')

model.load_state_dict(torch.load("/kaggle/input/efficientnet-pytorch/efficientnet-b2-27687264.pth"))



num_ftrs = model._fc.in_features



model._fc = nn.Linear(num_ftrs, 5)





#model.load_state_dict(torch.load("/kaggle/input/efficientnet-pytorch/efficientnet-b2-27687264.pth"))

model = model.to(device)

"""
"""

ct = 0

for child in model.children():

    ct += 1

    if ct < 3:

        for param in child.parameters():

            param.requires_grad = False

    else:

        for param in child.parameters():

            param.requires_grad = True

"""
def class_proportional_weights(train_labels):

    '''

    helper function to scale weights of classes in loss function based on their sampled proportions

    # This custom loss function is defined to reduce the effect of class imbalance.

    # Since there are so many samples labeled as "O", this allows the RNN to not 

    # be weighted too heavily in that area.

    '''

    weights = []

    for lab in range(0,5):

        weights.append(1-(train_labels.count(lab)/(len(train_labels)-train_labels.count(lab)))) #proportional to number without tags

    return weights
"""

weights = class_proportional_weights(train_pd.iloc[train_idx, 'diagnosis']) 

class_weights = torch.FloatTensor(weights).cuda()

criterion = nn.CrossEntropyLoss(weights=class_weights)"""
"""

from efficientnet_pytorch import EfficientNet



BATCH_SIZE = 28

train_pd = pd.read_csv(r"/kaggle/input/aptos2019-blindness-detection/train.csv")



def compute_accuracy(model, data_loader, device):

    correct_pred, num_examples = 0, 0

    tqdm()

    all_labels = []

    all_preds = []

    for i, (inputs, labels) in enumerate(tqdm(data_loader)):

        inputs = inputs.to(device)

        labels = labels.to(device)

        

        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)

        num_examples += labels.size(0)

        correct_pred += (preds == labels).sum()

        all_labels.extend([label.item() for label in labels])

        all_preds.extend([pred.item() for pred in preds])

    return np.array(all_labels), np.array(all_preds), correct_pred.float()/num_examples * 100







device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 5

N_SPLITS = 2

start_time = time.time()



splits = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=1234).split(train_pd['id_code'], train_pd['diagnosis']))





# using a numpy array because it's faster than a list

test_pd = pd.read_csv(r"/kaggle/input/aptos2019-blindness-detection/test.csv")



predictionsfinal = torch.zeros((len(test_pd),5), dtype=torch.float32)

test_dataset = APTOSDataset(csv_file='../input/aptos2019-blindness-detection/test.csv', filetype='test',transform=test_transform)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)







for i, (train_idx, valid_idx) in enumerate(splits):

    print("\n")

    print("--- Fold Number: {} ---".format(i+1))

    train_dataset = KFoldDataset(csv_file='../input/aptos2019-blindness-detection/train.csv', indices=train_idx,filetype='train',transform=train_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    valid_dataset = KFoldDataset(csv_file='../input/aptos2019-blindness-detection/train.csv', indices=valid_idx,filetype='train',transform=valid_transform)

    val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)



    



    model = EfficientNet.from_name('efficientnet-b3')

    model.load_state_dict(torch.load("/kaggle/input/efficientnet-pytorch/efficientnet-b3-c8376fa2.pth"))



    num_ftrs = model._fc.in_features



    model._fc = nn.Linear(num_ftrs, 5)

    #model.load_state_dict(torch.load("/kaggle/input/efficientnet-pytorch/efficientnet-b2-27687264.pth"))

    model = model.to(device)

    



    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    losses = []

    valid_acc_list = []

    best_acc = 0.0



    

    for epoch in range(NUM_EPOCHS):

        print('Seconds elapsed: ', round((time.time() - start_time),2))

        print('Running Epoch: ', epoch+1)

        print('-' * 10)

        running_loss = 0.0

        model.train()

        i = 0

        for iteration, (inputs, labels) in enumerate(train_loader):

            inputs = inputs.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

        

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += float(loss.item())

            del inputs, labels

            i+=1

            # printing the results every 20 iterations

            if not iteration%20:

                print('Epoch {:03d}/{:03d} | Batch: {:03d}/{:03d} |'

                      ' Cost: {:.4f} | Avg Loss: {:.4f}'.format(epoch+1, NUM_EPOCHS, iteration+1, len(train_loader),loss, running_loss/i))

                running_loss = 0.0

                i = 0

            del loss

            gc.collect()

            torch.cuda.empty_cache()

          

        with torch.set_grad_enabled(False):

            model.eval()

            train_labels, train_preds, train_accuracy = compute_accuracy(model, train_loader, device)

            valid_labels, valid_preds, valid_accuracy = compute_accuracy(model, val_loader, device)

            valid_acc_list.append(valid_accuracy)

            print('Epoch: {:03d}/{:03d} Train Acc.: {:.2f} | Validation Acc.: {:.2f}'.format(epoch+1,NUM_EPOCHS, train_accuracy,

                                                                                    valid_accuracy))

            print('Confusion Matrix for Train and Validation')

            train_cnf_matrix = confusion_matrix(train_labels, train_preds)

            valid_cnf_matrix = confusion_matrix(valid_labels, valid_preds)

            train_cnf_matrix_norm = train_cnf_matrix.astype('float') / train_cnf_matrix.sum(axis=1)[:, np.newaxis]

            validation_cnf_matrix_norm = valid_cnf_matrix.astype('float') / valid_cnf_matrix.sum(axis=1)[:, np.newaxis]

            graph_labels = ['0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative DR']

            train_df_cm = pd.DataFrame(train_cnf_matrix_norm, index=graph_labels, columns=graph_labels)

            validation_df_cm = pd.DataFrame(validation_cnf_matrix_norm, index=graph_labels, columns=graph_labels)

            fig, (ax1, ax2) = plt.subplots(1,2, sharex='col', figsize=(10,4))

            sns.heatmap(train_df_cm, annot=True, fmt='.2f', cmap="Blues", ax=ax1).set_title('Train')

            sns.heatmap(validation_df_cm, annot=True, fmt='.2f', cmap=sns.cubehelix_palette(8), ax=ax2).set_title('Validation')

            plt.show()

            print('-'*20)

            print('\n')



    kfold_test_predictions = torch.zeros((len(test_pd)),5)

    model.eval()

    with torch.no_grad():

        for i, (inputs, img_id) in enumerate(tqdm(test_loader)):

            inputs = inputs.to(device)

            outputs = model(inputs)

            kfold_test_predictions[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = outputs



        predictionsfinal += (kfold_test_predictions/N_SPLITS)

        

_, preds = torch.max(predictionsfinal, 1)     

final_predictions = pd.DataFrame(([test_pd['id_code'].tolist(), preds.tolist()])).transpose()

final_predictions.columns = ['id_code', 'diagnosis']

final_predictions_df = final_predictions.copy()

final_predictions.to_csv("submission.csv",index=False)

print("Training Complete --- {} seconds ---".format(round((time.time() - start_time),2)))

"""
"""

from tqdm._tqdm_notebook import tqdm_notebook

def compute_predictions(model, model_type, data_loader, device):

    if model_type == 'train':

        predictions = []

        correct_pred, num_examples = 0, 0

        tqdm_notebook()

        for i, (inputs, labels) in enumerate(tqdm_notebook(data_loader)):

            inputs = inputs.to(device)

            labels = labels.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            predictions.append(preds)

            num_examples += labels.size(0)

            correct_pred += (preds==labels).sum()

        return predictions, correct_pred.item()/num_examples*100

    

    else:

        predictions = []

        img_ids = []

        tqdm_notebook()

        for i, (inputs, img_id) in enumerate(tqdm_notebook(data_loader)):

            inputs = inputs.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            predictions.extend(preds)

            img_ids.extend(img_id)

            

        predictions = [pred.item() for pred in predictions]

        final_predictions = pd.DataFrame(np.array([img_ids, predictions])).transpose()

        final_predictions.columns = ['id_code', 'diagnosis']

        final_predictions_df = final_predictions.copy()

        final_predictions.to_csv("submission.csv",index=False)

        return final_predictions_df

            

"""            
"""

with torch.set_grad_enabled(False):

    model.eval()

    #train_predictions, train_accuracy = compute_predictions(model, 'train', train_loader, device)

    #print('Train Accuracy: ', train_accuracy)

    print('Computing Test Predictions')

    test_predictions = compute_predictions(model, 'test', test_loader, device)

        

"""    

    
from efficientnet_pytorch import EfficientNet



BATCH_SIZE = 28

train_pd = pd.read_csv(r"/kaggle/input/aptos2019-blindness-detection/train.csv")





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 5

N_SPLITS = 5

start_time = time.time()



splits = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=1234).split(train_pd['id_code'], train_pd['diagnosis']))





# using a numpy array because it's faster than a list

test_pd = pd.read_csv(r"/kaggle/input/aptos2019-blindness-detection/test.csv")



predictionsfinal = torch.zeros((len(test_pd),5), dtype=torch.float32)

test_dataset = APTOSDataset(csv_file='../input/aptos2019-blindness-detection/test.csv', filetype='test',transform=test_transform)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)







for fold_num, (train_idx, valid_idx) in enumerate(splits):

    print("\n")

    print("--- Fold Number: {} ---".format(fold_num+1))



    model = EfficientNet.from_name('efficientnet-b2')

    num_ftrs = model._fc.in_features

    model._fc = nn.Linear(num_ftrs, 5)

    model.load_state_dict(torch.load("/kaggle/input/pretrained-cv-weights/valid_fold_"+str(fold_num+1)+".pth"))

    model = model.to(device)





    kfold_test_predictions = torch.zeros((len(test_pd)),5)

    model.eval()

    with torch.no_grad():

        for i, (inputs, img_id) in enumerate(tqdm(test_loader)):

            inputs = inputs.to(device)

            outputs = model(inputs)

            kfold_test_predictions[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = outputs



        predictionsfinal += (kfold_test_predictions/N_SPLITS)

        

_, preds = torch.max(predictionsfinal, 1)     

final_predictions = pd.DataFrame(([test_pd['id_code'].tolist(), preds.tolist()])).transpose()

final_predictions.columns = ['id_code', 'diagnosis']

final_predictions_df = final_predictions.copy()

final_predictions.to_csv("submission.csv",index=False)

print("Training Complete --- {} seconds ---".format(round((time.time() - start_time),2)))