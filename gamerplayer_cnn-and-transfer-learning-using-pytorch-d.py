# import os

# import urllib.request
# !curl -o 'data.zip' 'https://storage.googleapis.com/kaggle-data-sets/721614%2F1255086%2Fbundle%2Farchive.zip?GoogleAccessId=gcp-kaggle-com@kaggle-161607.iam.gserviceaccount.com&Expires=1594111398&Signature=ZpADxafeoDZk4v%2BIzC6Vk9dDkbyO0Cit%2FUozBr2YiKPb29ecL2C1NKHOw%2BkMero7myp4wBZhY2M5wxjqK%2BNNQ1TBu936FeQbpNeIkXaqrQRVWVGhSCwYj7J1bW%2BO7Hx81TODP1Hltgq1cNeZKihGKaJzBRksUS%2FbGbjI8fxTTJjGJqwih2R4gsRbelQnAzfhiOMRa5EF0DI%2BrH%2BpTtQTdVG1V15MDYt9JLmz2w59iJO9DxGodJvQv287XXuyKxMfYAOCnCu1qtiE%2FFf%2BO4G6ztaN4mOop0jxyRYD6Q7fXq4tYc4Dtw%2BHJIqsxTVOPQX%2BoDkgFr6%2BDVFurOldiPtSDg%3D%3D' -H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Accept-Language: en-US,en;q=0.5' --compressed -H 'Referer: https://www.kaggle.com/' -H 'Connection: keep-alive' -H 'Upgrade-Insecure-Requests: 1' -H 'TE: Trailers'
# os.listdir()
# !unzip data.zip

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('images'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torchvision 

from torchvision import transforms

import torch as T

T.manual_seed(0)


image_transforms = {

    # Train uses data augmentation

    'train':

    transforms.Compose([

        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),

        transforms.RandomRotation(degrees=15),

        transforms.ColorJitter(),

        transforms.RandomHorizontalFlip(),

        transforms.CenterCrop(size=224),  # Image net standards

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406],

                             [0.229, 0.224, 0.225])  # Imagenet standards

    ]),

   

    }
imagef.class_to_idx
imagef = torchvision.datasets.ImageFolder('../input/freibergs-groceries/images', transform=image_transforms['train'])



print(imagef)

print(imagef.classes)

img, label = imagef[0]

display(img)

print(img.size())

print(label)
print(imagef)

print(imagef.classes)

img, label = imagef[3]

print(img.size())

print(label)
imagef.classes
len(imagef.classes)
import matplotlib.pyplot as plt
plt.imshow(img.T)
from torch.utils.data import DataLoader
batch_size=128
data = {

    'train':imagef

}



# Dataloader iterators, make sure to shuffle

dataloaders = {

    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),

}
dataloaders['train'].dataset
import torch as T

import torchvision as tv
from torchvision import models

model = models.vgg19(pretrained=True)
for param in model.parameters():

    param.requires_grad = False
import torch.nn as nn
model
model.classifier
model.classifier[6] = nn.Sequential(

                      nn.Linear(4096, 2048), 

                      nn.ReLU(), 

                      nn.Dropout(0.4),

                      nn.Linear(2048, 1024),

                      nn.ReLU(),

                      nn.Dropout(0.2),

                      nn.Linear(1024, 512),

                      nn.ReLU(),

                      nn.Dropout(0.1),

                      nn.Linear(512,len(imagef.classes)),                   

                      nn.LogSoftmax(dim=1))
model
# Move to gpu

model = model.to('cuda')

# Distribute across 2 gpus

model = nn.DataParallel(model)
from torch import optim# Loss and optimizer

criteration = nn.CrossEntropyLoss().to('cuda')

optimizer = optim.Adam(model.parameters())

losses=[]

max_loss=[]

for epoch in range(100):

  losss=0

  for data, targets in dataloaders['train']:

    # Generate predictions

    out = model(data.to('cuda'))

    optimizer.zero_grad()

    # Calculate loss

    loss = criteration(out, targets.to('cuda'))

    # Backpropagation

    loss.backward()

    # Update model parameters

    optimizer.step()

    losses.append(loss.item())

    losss+=loss.item()

    print('',end='.')

  max_loss.append((losss/float(30))) 

  if epoch%10 == 0 :

        print('\n\nloss is {}'.format((losss/float(30))))
def predict(model,data):

    logs=model(data.to('cuda'))

    p=T.exp(logs)

    pred = T.max(p, dim=1)

    return pred
model.eval()
acc=0

length=0

for data,targets in dataloaders['train']:

    y_hat=predict(model2,data)

    acc+=(y_hat[1].to('cpu')==targets).sum()

    length+=len(targets)
acc
a=(targets==y_hat[1].to('cpu'))
acc/float(length)
plt.scatter([ x for x in range(len(losses))],losses)
T.save(model,'/content/drive/My Drive/class_prjct_weights/model.pt')
import torch as T
from google.colab import drive

drive.mount('/content/drive')
model2=T.load('/content/drive/My Drive/class_prjct_weights/model.pt')
model2
acc=0

length=0

for data,targets in dataloaders['train']:

    y_hat=predict(model2,data)

    acc+=(y_hat[1].to('cpu')==targets).sum()

    length+=len(targets)
acc.item()/length
targets
y_hat[1]
model2.to('cpu')
from sklearn.metrics import confusion_matrix
confusion_matrix(y_hat[1],targets)