!pip install efficientnet_pytorch



from efficientnet_pytorch import EfficientNet



import torch

import torch.utils.data as Data

import torch.nn as nn

from torchvision import transforms, models

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from PIL import Image

from sklearn.metrics import accuracy_score, confusion_matrix

from scipy.special import softmax

import cv2

from transformers import get_cosine_schedule_with_warmup

from transformers import AdamW

from tqdm.notebook import tqdm

from albumentations import *

from albumentations.pytorch import ToTensor

import gc



import warnings

warnings.filterwarnings("ignore")
im_healthy = plt.imread('../input/plant-pathology-2020-fgvc7/images/Train_2.jpg', format = 'jpg')

im_multi = plt.imread('../input/plant-pathology-2020-fgvc7/images/Train_1.jpg', format = 'jpg')

im_rust = plt.imread('../input/plant-pathology-2020-fgvc7/images/Train_3.jpg', format = 'jpg')

im_scab = plt.imread('../input/plant-pathology-2020-fgvc7/images/Train_0.jpg', format = 'jpg')



fig = plt.figure(figsize=(16,10))

ax = fig.add_subplot(2, 2, 1)

ax.imshow(im_healthy)

ax.set_title('Healthy', fontsize = 20)



ax = fig.add_subplot(2, 2, 2)

ax.imshow(im_multi)

ax.set_title('Multiple Diseases', fontsize = 20)



ax = fig.add_subplot(2, 2, 3)

ax.imshow(im_rust)

ax.set_title('Rust', fontsize = 20)



ax = fig.add_subplot(2, 2, 4)

ax.imshow(im_scab)

ax.set_title('Scab', fontsize = 20)
IMAGE_FOLDER = '../input/plant-pathology-npy-images/kaggle/working/image_pickles/'



def get_image_path(filename):

    return (IMAGE_FOLDER + filename + '.npy')
train = pd.read_csv('../input/plant-pathology-2020-fgvc7/train.csv')

test = pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')



train['image_path'] = train['image_id'].apply(get_image_path)

test['image_path'] = test['image_id'].apply(get_image_path)

train_labels = train.loc[:, 'healthy':'scab']

train_paths = train.image_path

test_paths = test.image_path
from sklearn.model_selection import train_test_split



train_paths, valid_paths, train_labels, valid_labels = train_test_split(train_paths, train_labels, test_size = 0.2, random_state=23, stratify = train_labels)

train_paths.reset_index(drop=True,inplace=True)

train_labels.reset_index(drop=True,inplace=True)

valid_paths.reset_index(drop=True,inplace=True)

valid_labels.reset_index(drop=True,inplace=True)
class LeafDataset(Data.Dataset):

    def __init__(self, image_paths, labels = None, train = True, test = False):

        self.paths = image_paths

        self.test = test

        if self.test == False:

            self.labels = labels

        self.train = train

        self.train_transform = Compose([HorizontalFlip(p=0.5),

                                  VerticalFlip(p=0.5),

                                  ShiftScaleRotate(rotate_limit=25.0, p=0.7),

                                  OneOf([IAAEmboss(p=1),

                                         IAASharpen(p=1),

                                         Blur(p=1)], p=0.5),

                                  IAAPiecewiseAffine(p=0.5)])

        self.test_transform = Compose([HorizontalFlip(p=0.5),

                                       VerticalFlip(p=0.5),

                                       ShiftScaleRotate(rotate_limit=25.0, p=0.7)])

        self.default_transform = Compose([Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), always_apply=True),

                                         ToTensor()]) #normalized for pretrained network

        

    def __len__(self):

        return self.paths.shape[0]

    

    def __getitem__(self, i):

        image = np.load(self.paths[i]) #load from .npy file!

        if self.test==False:

            label = torch.tensor(np.argmax(self.labels.loc[i,:].values)) #loss function used later doesnt take one-hot encoded labels, so convert it using argmax

        if self.train:

            image = self.train_transform(image=image)['image']

            image = self.default_transform(image=image)['image']

        elif self.test:

            image = self.test_transform(image=image)['image']

            image = self.default_transform(image=image)['image']

        else:

            image = self.default_transform(image=image)['image']

        

        if self.test==False:

            return image, label

        return image
def train_fn(net, loader):

    

    running_loss = 0

    preds_for_acc = []

    labels_for_acc = []

    

    pbar = tqdm(total = len(loader), desc='Training')

    

    for _, (images, labels) in enumerate(loader):

        

        images, labels = images.to(device), labels.to(device)

        net.train()

        optimizer.zero_grad()

        predictions = net(images)

        loss = loss_fn(predictions, labels)

        loss.backward()

        optimizer.step()

        scheduler.step()

        

        running_loss += loss.item()*labels.shape[0]

        labels_for_acc = np.concatenate((labels_for_acc, labels.cpu().numpy()), 0)

        preds_for_acc = np.concatenate((preds_for_acc, np.argmax(predictions.cpu().detach().numpy(), 1)), 0)

        

        pbar.update()

        

    accuracy = accuracy_score(labels_for_acc, preds_for_acc)

    

    pbar.close()

    return running_loss/TRAIN_SIZE, accuracy



def valid_fn(net, loader):

    

    running_loss = 0

    preds_for_acc = []

    labels_for_acc = []

    

    pbar = tqdm(total = len(loader), desc='Validation')

    

    with torch.no_grad():       #torch.no_grad() prevents Autograd engine from storing intermediate values, saving memory

        for _, (images, labels) in enumerate(loader):

            

            images, labels = images.to(device), labels.to(device)

            net.eval()

            predictions = net(images)

            loss = loss_fn(predictions, labels)

            

            running_loss += loss.item()*labels.shape[0]

            labels_for_acc = np.concatenate((labels_for_acc, labels.cpu().numpy()), 0)

            preds_for_acc = np.concatenate((preds_for_acc, np.argmax(predictions.cpu().detach().numpy(), 1)), 0)

            

            pbar.update()

            

        accuracy = accuracy_score(labels_for_acc, preds_for_acc)

        conf_mat = confusion_matrix(labels_for_acc, preds_for_acc)

    

    pbar.close()

    return running_loss/VALID_SIZE, accuracy, conf_mat



def test_fn(net, loader):



    preds_for_output = np.zeros((1,4))

    

    with torch.no_grad():

        pbar = tqdm(total = len(loader))

        for _, images in enumerate(loader):

            images = images.to(device)

            net.eval()

            predictions = net(images)

            preds_for_output = np.concatenate((preds_for_output, predictions.cpu().detach().numpy()), 0)

            pbar.update()

    

    pbar.close()

    return preds_for_output
BATCH_SIZE = 8

NUM_EPOCHS = 30

TRAIN_SIZE = train_labels.shape[0]

VALID_SIZE = valid_labels.shape[0]

MODEL_NAME = 'efficientnet-b5'

device = 'cuda'

lr = 8e-4
train_dataset = LeafDataset(train_paths, train_labels)

trainloader = Data.DataLoader(train_dataset, shuffle=True, batch_size = BATCH_SIZE, num_workers = 2)



valid_dataset = LeafDataset(valid_paths, valid_labels, train = False)

validloader = Data.DataLoader(valid_dataset, shuffle=False, batch_size = BATCH_SIZE, num_workers = 2)



test_dataset = LeafDataset(test_paths, train = False, test = True)

testloader = Data.DataLoader(test_dataset, shuffle=False, batch_size = BATCH_SIZE, num_workers = 2)
model = EfficientNet.from_pretrained(MODEL_NAME)



num_ftrs = model._fc.in_features

model._fc = nn.Sequential(nn.Linear(num_ftrs,1000,bias=True),

                          nn.ReLU(),

                          nn.Dropout(p=0.5),

                          nn.Linear(1000,4, bias = True))



model.to(device)

optimizer = AdamW(model.parameters(), lr = lr, weight_decay = 1e-3)

num_train_steps = int(len(train_dataset) / BATCH_SIZE * NUM_EPOCHS)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataset)/BATCH_SIZE*5, num_training_steps=num_train_steps)

loss_fn = torch.nn.CrossEntropyLoss()
train_loss = []

valid_loss = []

train_acc = []

val_acc = []
for epoch in range(NUM_EPOCHS):

    

    tl, ta = train_fn(model, loader = trainloader)

    vl, va, conf_mat = valid_fn(model, loader = validloader)

    train_loss.append(tl)

    valid_loss.append(vl)

    train_acc.append(ta)

    val_acc.append(va)

    

    if (epoch+1)%10==0:

        path = 'epoch' + str(epoch) + '.pt'

        torch.save(model.state_dict(), path)

    

    printstr = 'Epoch: '+ str(epoch) + ', Train loss: ' + str(tl) + ', Val loss: ' + str(vl) + ', Train acc: ' + str(ta) + ', Val acc: ' + str(va)

    tqdm.write(printstr)

    

'''Output hidden. Unhide to see training log'''
plt.figure()

plt.ylim(0,1.5)

sns.lineplot(list(range(len(train_loss))), train_loss)

sns.lineplot(list(range(len(valid_loss))), valid_loss)

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend(['Train','Val'])
plt.figure()

sns.lineplot(list(range(len(train_acc))), train_acc)

sns.lineplot(list(range(len(val_acc))), val_acc)

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(['Train','Val'])
labels = ['Healthy', 'Multiple','Rust','Scab']

sns.heatmap(conf_mat, xticklabels=labels, yticklabels=labels, annot=True)
subs = []

for i in range(5): #average over 5 runs

    out = test_fn(model, testloader)

    output = pd.DataFrame(softmax(out,1), columns = ['healthy','multiple_diseases','rust','scab']) #the submission expects probability scores for each class

    output.drop(0, inplace = True)

    output.reset_index(drop=True,inplace=True)

    subs.append(output)



sub_eff1 = sum(subs)/5
sub1 = sub_eff1.copy()

sub1['image_id'] = test.image_id

sub1 = sub1[['image_id','healthy','multiple_diseases','rust','scab']]

sub1.to_csv('submission_efficientnet1.csv', index = False)
del model

del optimizer

del scheduler

torch.cuda.empty_cache()
model = EfficientNet.from_pretrained(MODEL_NAME)



num_ftrs = model._fc.in_features

model._fc = nn.Sequential(nn.Linear(num_ftrs,1000,bias=True),

                          nn.ReLU(),

                          nn.Dropout(p=0.5),

                          nn.Linear(1000,4, bias = True))



model.to(device)

optimizer = AdamW(model.parameters(), lr = lr, weight_decay = 1e-3)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataset)/BATCH_SIZE*5, num_training_steps=num_train_steps)
train_loss = []

valid_loss = []

train_acc = []

val_acc = []
for epoch in range(NUM_EPOCHS):

    

    tl, ta = train_fn(model, loader = trainloader)

    vl, va, conf_mat = valid_fn(model, loader = validloader)

    train_loss.append(tl)

    valid_loss.append(vl)

    train_acc.append(ta)

    val_acc.append(va)

    

    if (epoch+1)%10==0:

        path = 'epoch' + str(epoch) + '.pt'

        torch.save(model.state_dict(), path)

    

    printstr = 'Epoch: '+ str(epoch) + ', Train loss: ' + str(tl) + ', Val loss: ' + str(vl) + ', Train acc: ' + str(ta) + ', Val acc: ' + str(va)

    tqdm.write(printstr)

    

'''Output hidden. Unhide to see training log'''
plt.figure()

plt.ylim(0,1.5)

sns.lineplot(list(range(len(train_loss))), train_loss)

sns.lineplot(list(range(len(valid_loss))), valid_loss)

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend(['Train','Val'])
plt.figure()

sns.lineplot(list(range(len(train_acc))), train_acc)

sns.lineplot(list(range(len(val_acc))), val_acc)

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(['Train','Val'])
labels = ['Healthy', 'Multiple','Rust','Scab']

sns.heatmap(conf_mat, xticklabels=labels, yticklabels=labels, annot=True)
subs = []

for i in range(5): #average over 5 runs

    out = test_fn(model, testloader)

    output = pd.DataFrame(softmax(out,1), columns = ['healthy','multiple_diseases','rust','scab']) #the submission expects probability scores for each class

    output.drop(0, inplace = True)

    output.reset_index(drop=True,inplace=True)

    subs.append(output)



sub_eff2 = sum(subs)/5
sub2 = sub_eff2.copy()

sub2['image_id'] = test.image_id

sub2 = sub2[['image_id','healthy','multiple_diseases','rust','scab']]

sub2.to_csv('submission_efficientnet2.csv', index = False)
sub = (sub_eff1 + sub_eff2)/2

sub['image_id'] = test.image_id

sub = sub[['image_id','healthy','multiple_diseases','rust','scab']]

sub.to_csv('submission.csv', index = False)