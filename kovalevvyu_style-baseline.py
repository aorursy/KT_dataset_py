# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/painter-by-numbers-resized/artist_info_full.csv')

df.head()
TOP_N = 5
df['style'].value_counts()[0:TOP_N]
style_enc_dict = {}

for i, style in enumerate(df['style'].value_counts()[0:TOP_N].index.tolist()):

    style_enc_dict[style] = i

print(style_enc_dict)
def get_key(my_dict, val): 

    for key, value in my_dict.items(): 

        if val == value:

            return key
style_df = df[df['style'].isin(style_enc_dict.keys())]
style_df.head()
style_df['style'].value_counts()
style_df['style'].update(style_df['style'].map(style_enc_dict))
style_df['style'].value_counts()
sum(style_df['style'].value_counts())
from sklearn.model_selection import train_test_split
y = style_df['style']

y.head()
style_df.drop(labels=['style'], axis=1, inplace=True)

style_df.head()
X = style_df
X_part, X_not_used, y_part, y_not_used = train_test_split(\

                                                    X, y,\

                                                    test_size=0.75, shuffle=True,\

                                                    stratify = y, random_state=42)
print(X_part.shape,y_part.shape)
X_train, X_test, y_train, y_test = train_test_split(\

                                                    X_part, y_part,\

                                                    test_size=0.20, shuffle=True,\

                                                    stratify = y_part, random_state=42)
print(X_train.shape,y_train.shape)
X_train.head()
y_train.value_counts()
y_test.value_counts()
X_train.head()
train_df = X_train.join(y_train)

train_df.head()
test_df = X_test.join(y_test)

test_df.head()
import torch

import numpy as np

import torchvision

import matplotlib.pyplot as plt

import time

import copy

from PIL import Image

from torchvision import transforms, models

from torch.utils.data.sampler import SubsetRandomSampler



import random

import shutil 

import cv2

from tqdm import tqdm
!pip install knockknock
from knockknock import telegram_sender



CHAT_ID: int = 266478885

@telegram_sender(token="647225942:AAF-biI_UdXDVOwhqBjFRcELwbTzdeidn0w", chat_id=CHAT_ID)

def train_your_nicest_model(time_value=2):

    import time

    time.sleep(time_value)

    return {'loss': 0.9} # Optional return value
train_your_nicest_model()
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

torch.cuda.manual_seed(RANDOM_SEED)

torch.backends.cudnn.deterministic = True
filepath = '../input/painter-by-numbers-resized/'

class ImagesDataset(torch.utils.data.Dataset):

    def __init__(self, df, transform = None,

                 loader = torchvision.datasets.folder.default_loader):

        self.df = df

        self.transform = transform

        self.loader = loader



    def __getitem__(self, index):

        row = self.df.iloc[index]

        target = row['style']

        path = filepath + row['filename']

        img = self.loader(path)

        if self.transform is not None:

            img = self.transform(img)

        return img, target



    def __len__(self):

        n, _ = self.df.shape

        return n
# what transformations should be done with our images

train_transforms = transforms.Compose([

    transforms.RandomAffine(degrees=10, scale=(1.1, 1.3)),

    transforms.RandomCrop((224, 224)),

    transforms.RandomHorizontalFlip(),

#     transforms.RandomVerticalFlip(),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])



val_transforms = transforms.Compose([

    transforms.RandomCrop((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])
# initialize our dataset at first

dataset = ImagesDataset(

    df = train_df,

    transform = train_transforms

)



batch_size = 16

validation_split = 0.2

shuffle_dataset = True

# Creating data indices for training and validation splits:

dataset_size = len(train_df)



indices = list(range(dataset_size))

split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset:

    np.random.seed(RANDOM_SEED)

    np.random.shuffle(indices)



train_indices, val_indices = indices[split:], indices[:split]



# Creating PT data samplers and loaders:

train_sampler = SubsetRandomSampler(train_indices)

valid_sampler = SubsetRandomSampler(val_indices)



train_dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, 

                                           sampler = train_sampler)

val_dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size,

                                                sampler = valid_sampler)
len(train_dataloader), len(train_indices)
X_batch, y_batch = next(iter(train_dataloader))

mean = np.array([0.485, 0.456, 0.406])

std = np.array([0.229, 0.224, 0.225])

plt.imshow(X_batch[0].permute(1, 2, 0).numpy() * std + mean);

plt.title(get_key(style_enc_dict, int(y_batch[0])))

plt.show()
def show_input(input_tensor, title=''):

    image = input_tensor.permute(1, 2, 0).numpy()

    image = std * image + mean

    

    plt.imshow(image.clip(0, 1))

    plt.title(title)

    plt.show()

    plt.pause(0.001)



X_batch, y_batch = next(iter(train_dataloader))



iter_stop = 10

i = 0



for x_item, y_item in zip(X_batch, y_batch):

    show_input(x_item, title=get_key(style_enc_dict, y_item))

    i += 1

    if i > iter_stop:

        break
# def f1_loss(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:

#     '''Calculate F1 score. Can work with gpu tensors

    

#     The original implmentation is written by Michal Haltuf on Kaggle.

    

#     Returns

#     -------

#     torch.Tensor

#         `ndim` == 1. 0 <= val <= 1

    

#     Reference

#     ---------

#     - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric

#     - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score

#     - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    

#     '''

#     assert y_true.ndim == 1

#     assert y_pred.ndim == 1 or y_pred.ndim == 2

    

#     if y_pred.ndim == 2:

#         y_pred = y_pred.argmax(dim=1)

        

    

#     tp = (y_true * y_pred).sum().to(torch.float32)

#     tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)

#     fp = ((1 - y_true) * y_pred).sum().to(torch.float32)

#     fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    

#     epsilon = 1e-7

    

#     precision = tp / (tp + fp + epsilon)

#     recall = tp / (tp + fn + epsilon)

    

#     f1 = 2* (precision*recall) / (precision + recall + epsilon)

#     f1.requires_grad = is_training

#     return f1
@telegram_sender(token="647225942:AAF-biI_UdXDVOwhqBjFRcELwbTzdeidn0w", chat_id=CHAT_ID)

def model_print(epoch, num_epochs):

    return "Epoch: " + str(epoch) + "/" + str(num_epochs)
train_accuracy_history = []

train_loss_history = []

# train_roc_auc_history = []





val_accuracy_history = []

val_loss_history = []

# val_roc_auc_history = []



res_model = None

@telegram_sender(token="647225942:AAF-biI_UdXDVOwhqBjFRcELwbTzdeidn0w", chat_id=CHAT_ID)

def train_model(model, loss, optimizer, num_epochs):

    min_val_loss = 200.0

    for epoch in range(num_epochs):

        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)

        model_print(epoch, num_epochs - 1)

        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:

            if phase == 'train':

                dataloader = train_dataloader

                scheduler.step()

                model.train()  # Set model to training mode

            else:

                dataloader = val_dataloader

                model.eval()   # Set model to evaluate mode



            running_loss = 0.

            running_acc = 0.

            # running_roc_auc = 0.

            

            

            # Iterate over data.

            for inputs, labels in tqdm(dataloader):

                inputs = inputs.to(device)

                labels = labels.to(device)



                optimizer.zero_grad()



                # forward and backward

                with torch.set_grad_enabled(phase == 'train'):

                    preds = model(inputs)

                    loss_value = loss(preds, labels)

                    preds_class = preds.argmax(dim=1)



                    # backward + optimize only if in training phase

                    if phase == 'train':

                        loss_value.backward()

                        optimizer.step()



                # statistics

                running_loss += loss_value.item()

                running_acc += (preds_class == labels.data).float().mean()

                # running_roc_auc += f1_loss(labels.data, preds_class).mean()

                

            epoch_loss = running_loss / len(dataloader)

            epoch_acc = running_acc / len(dataloader)

            # epoch_roc_auc = running_roc_auc / len(dataloader)

            

            if (phase == 'train'):

                train_accuracy_history.append(epoch_acc)

                train_loss_history.append(epoch_loss)

                # train_roc_auc_history.append(epoch_roc_auc)

                

            elif (phase == 'val'):

                if epoch_loss < min_val_loss:

                    min_val_loss = epoch_loss

                    with open('top_model', 'wb') as f:

                        torch.save(model, f)

                val_accuracy_history.append(epoch_acc)

                val_loss_history.append(epoch_loss)

                # val_roc_auc_history.append(epoch_roc_auc)

                

            print('{} loss: {:.4f} acc: {:.4f}'.format(phase, epoch_loss, \

                                                       epoch_acc), flush=True)

    print('top model with min_val_loss:', min_val_loss)

    return 'Success'
model = models.resnet50(pretrained=True)



# Disable grad for all conv layers

# for param in model.parameters():

#     t = param

#     param.requires_grad = False



num_ftrs = model.fc.in_features

model.fc = torch.nn.Linear(num_ftrs, TOP_N)

# model.bn = torch.nn.BatchNorm1d(num_ftrs)

# model.fc = torch.nn.Linear(num_ftrs, int(num_ftrs / 2))

# num_ftrs = model.fc.in_features



# model.act1 = torch.nn.LeakyReLU()

# model.fc2 = torch.nn.Linear(num_ftrs, TOP_N)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)



loss = torch.nn.CrossEntropyLoss()



optimizer = torch.optim.Adam(model.parameters(), amsgrad=True, lr=5.0e-3)



# Decay LR by a factor of 0.1 every 10 epochs

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)
train_model(model, loss, optimizer, num_epochs=20);
plt.plot(train_accuracy_history, label='train_acc')

plt.plot(val_accuracy_history, label='val_acc')

plt.legend()

plt.title('Accuracy');
plt.plot(train_loss_history, label='train_loss')

plt.plot(val_loss_history, label='val_loss')

plt.legend()

plt.title('Loss');
# plt.plot(train_roc_auc_history, label='train_f1')

# plt.plot(val_roc_auc_history, label='val_f1')

# plt.legend()

# plt.title('f1');
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(\

                                                    X, y,\

                                                    test_size = 0.20, shuffle = True,\

                                                    stratify = y, random_state = RANDOM_SEED)
test_df_full = X_test_full.join([y_test_full])

test_df_full.head()
dataset = ImagesDataset(

    df = test_df_full,

    transform = val_transforms

)



test_dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size)
X_batch, y_batch = next(iter(test_dataloader))

mean = np.array([0.485, 0.456, 0.406])

std = np.array([0.229, 0.224, 0.225])

plt.imshow(X_batch[0].permute(1, 2, 0).numpy() * std + mean);

plt.title(get_key(style_enc_dict, int(y_batch[0])))

plt.show()
# Load the best saved model.

with open('top_model', 'rb') as f:

    model = torch.load(f)

model.eval()



test_labels = []

test_predictions = []

test_predictions_class = []

test_batch_loss = 0.0

test_batch_acc = 0.0



for inputs, labels in tqdm(test_dataloader):

    inputs = inputs.to(device)

    for element in labels:

        test_labels.append(int(element))

    labels = labels.to(device)

    with torch.set_grad_enabled(False):

        preds = model(inputs)

        loss_value = loss(preds, labels)

        preds_class = preds.argmax(dim=1)

        for element in preds_class:

            test_predictions_class.append(int(element))

    test_batch_loss += loss_value.item()

    test_batch_acc += (preds_class == labels.data).float().mean()

    test_predictions.append(

        torch.nn.functional.softmax(preds, dim=1)[:,1].data.cpu().numpy())

    

test_predictions = np.concatenate(test_predictions)



test_loss = test_batch_loss / len(test_dataloader)

test_acc = test_batch_acc / len(test_dataloader)
print('test_loss:', test_loss)

print('test_acc:', float(test_acc))
from IPython.display import FileLink

FileLink(r'top_model')