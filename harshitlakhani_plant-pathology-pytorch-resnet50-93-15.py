import time
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import OrderedDict


import cv2
import torch
from torch import optim
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from torchvision import models,transforms
import time

#changes
import albumentations as A
from albumentations.pytorch import ToTensorV2
root = "/kaggle/input/plant-pathology-2020-fgvc7/"
train = pd.read_csv(os.path.join(root,"train.csv"))
test = pd.read_csv(os.path.join(root,"test.csv"))
submission = pd.read_csv(os.path.join(root,"sample_submission.csv"))
images = os.path.join(root,"images")
diseases = dict()

for column in ["healthy","multiple_diseases","rust","scab"]:
    counts = pd.DataFrame(train[column].value_counts())
    diseases[column] = counts.iloc[1,0]
    
#bar chart to show different diseases    
fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(15,25))
ax1.bar(diseases.keys(),diseases.values(), color=["#6666ff"])
ax1.set_title('Bar Chart', fontsize=18)

ax2.pie(diseases.values(),labels = diseases.keys(), colors=["#6666ff","#4da6ff","#1ac6ff","#c44dff"], autopct='%1.1f%%')
ax2.set_title('Pie Chart', fontsize=18)
ax2.axis('equal') 

plt.show()

def ShowImages(images, typ):
    fig = figure(figsize=(16,12))
    number_of_images = len(images)
    for i in range(number_of_images):
        a=fig.add_subplot(1,number_of_images,i+1)
        a.set_title(typ, fontsize = 10)
        image = imread(os.path.join(root,"images",images[i]))
        imshow(image)
        axis('off')
        
col=["healthy","multiple_diseases","rust","scab"]
print("Row's are in order of", col)

for column in col:
    images = (train[train[column].apply(lambda x: x == 1)]["image_id"].sample(4).values) + ".jpg"
    ShowImages(images, column)
def get_path(image):
    return os.path.join(root,"images",image + ".jpg")

train_data = train.copy()
train_data["image_path"] = train_data["image_id"].apply(get_path)
train_labels = train.loc[:, "healthy":"scab"]

test_data = test.copy()
test_data["image_path"] = test_data["image_id"].apply(get_path)
test_paths = test_data["image_path"]

train_paths, valid_paths, train_labels, valid_labels = train_test_split(train_data["image_path"], train_labels, test_size = 0.2, random_state=23, stratify = train_labels)

train_paths.reset_index(drop=True,inplace=True)
train_labels.reset_index(drop=True,inplace=True)
valid_paths.reset_index(drop=True,inplace=True)
valid_labels.reset_index(drop=True,inplace=True)
mytransform = {
    "train": A.Compose([
    A.RandomResizedCrop(height=256, width=256, p=1.0),
    A.Flip(),
    A.ShiftScaleRotate(rotate_limit=1.0, p=0.8),
    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
    ]),
    "validation": A.Compose([
    A.RandomResizedCrop(height=256, width=256, p=1.0),
    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
    ]),
}


class ImageDataset(Data.Dataset):
    def __init__(self, images_path, labels = None , test=False, transform=None):
        self.images_path = images_path
        self.test = test
        if self.test == False:
            self.labels = labels
            
        self.images_transform = transform

    def __getitem__(self, index):
        if self.test == False:
            labels = torch.tensor(np.argmax(self.labels.iloc[index, :]))
        
        image = cv2.imread(self.images_path[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_transformed = self.images_transform(image=image)
        
        if self.test ==False:
            return image_transformed["image"], labels
        return image_transformed["image"]

    def __len__(self):
        return self.images_path.shape[0]
def train_function(model, loader):
    
    running_loss = 0
    preds_for_acc = []
    labels_for_acc = []
    
    progress = tqdm(loader, desc="Training")
    
    for _, (images,labels) in enumerate(progress):
        
        images, labels = images.to(device), labels.to(device)
        model.train()
        
        #optimizer
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()


        running_loss += loss.item()*labels.shape[0]
        labels_for_acc = np.concatenate((labels_for_acc,labels.cpu().detach().numpy()), axis=0)
        preds_for_acc = np.concatenate((preds_for_acc,np.argmax(predictions.cpu().detach().numpy(), axis=1)), axis=0)


    accuracy = accuracy_score(labels_for_acc, preds_for_acc)

    return running_loss/TRAIN_SIZE, accuracy

def valid_function(model, loader):
    running_loss = 0
    preds_for_acc = []
    labels_for_acc = []
    
    progress = tqdm(loader, desc="Validation")
    
    for _, (images,labels) in enumerate(progress):
        
        images, labels = images.to(device), labels.to(device)
        
        
        with torch.no_grad():
            model.eval()
            predictions = model(images)
        loss = loss_function(predictions, labels)

        running_loss += loss.item()*labels.shape[0]
        labels_for_acc = np.concatenate((labels_for_acc,labels.cpu().detach().numpy()), axis=0)
        preds_for_acc = np.concatenate((preds_for_acc,np.argmax(predictions.cpu().detach().numpy(), axis=1)), axis=0)


    accuracy = accuracy_score(labels_for_acc, preds_for_acc)
    conf_matrix = confusion_matrix(labels_for_acc, preds_for_acc)

    return running_loss/VALID_SIZE, accuracy, conf_matrix
BATCH_SIZE = 64 #4
NUM_EPOCHS = 15 #10
device = "cuda"
TRAIN_SIZE = train_labels.shape[0]
VALID_SIZE = valid_labels.shape[0]
learning_rate = 5e-5
train_images = ImageDataset(images_path=train_paths, labels=train_labels, transform=mytransform["train"])
train_loader = Data.DataLoader(train_images, shuffle=True, batch_size = BATCH_SIZE)

valid_images = ImageDataset(images_path=valid_paths, labels=valid_labels, transform=mytransform["validation"])
valid_loader = Data.DataLoader(valid_images, shuffle=False, batch_size = BATCH_SIZE)
model = models.resnet50(pretrained = True)

# Freeze training for all layers
for param in model.parameters():
    param.require_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_ftrs,512,bias=True),
                          nn.ReLU(),
                          nn.Dropout(p=0.3),
                          nn.Linear(512,4, bias = True))

model.to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
train_loss = []
valid_loss = []
train_acc = []
val_acc = []
for epoch in range(NUM_EPOCHS):
    
    tl, ta = train_function(model, loader = train_loader)
    vl, va, conf_mat = valid_function(model, loader = valid_loader)
    train_loss.append(tl)
    valid_loss.append(vl)
    train_acc.append(ta)
    val_acc.append(va)
    
    printstr = 'Epoch: '+ str(epoch) + ', Train loss: ' + str(tl) + ', Val loss: ' + str(vl) + ', Train acc: ' + str(ta) + ', Val acc: ' + str(va)
    tqdm.write(printstr)
import matplotlib.pyplot as plt
import seaborn as sns

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
def sigmoid(X):
    return 1/(1+np.exp(-X))

def test_function(model, loader):
    preds_for_output = np.zeros((1,4))
    
    progress = tqdm(loader, desc="Testing")
    with torch.no_grad():
        for _, images in enumerate(progress):
            images = images.to(device)
            model.eval()
            predictions = model(images)
            preds_for_output = np.concatenate((preds_for_output, sigmoid(predictions.cpu().detach().numpy())), axis=0)
        preds_for_output = np.delete(preds_for_output, 0, 0)
    return preds_for_output
test_images = ImageDataset(images_path=test_paths, test=True, transform=mytransform["validation"])
test_loader = Data.DataLoader(test_images, shuffle=False, batch_size = BATCH_SIZE)

predictions = test_function(model, test_loader)

submission[["healthy","multiple_diseases","rust","scab"]]  = predictions
submission.to_csv("submission_2.csv", index=False)