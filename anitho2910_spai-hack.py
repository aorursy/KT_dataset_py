# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/hackathon-blossom-flower-classification"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import torch

from torch import nn

import torch.nn.functional as F

from torchvision import datasets,transforms,models

from torch import optim

%matplotlib inline
train_transform=transforms.Compose([transforms.RandomRotation(20),

                                   transforms.Resize((224,224)),

                                   transforms.ToTensor(),

                                   transforms.Normalize([0.485,0.456,0.406],

                                                       [0.229,0.224,0.225])])

test_transform=transforms.Compose([transforms.Resize((224,224)),

                                   transforms.ToTensor(),

                                   transforms.Normalize([0.485,0.456,0.406],

                                                       [0.229,0.224,0.225])])
dataTrain=datasets.ImageFolder('../input/hackathon-blossom-flower-classification/flower_data/flower_data/train',transform=train_transform)

dataTest=datasets.ImageFolder('../input/hackathon-blossom-flower-classification/flower_data/flower_data/valid',transform=train_transform)

trainLoader=torch.utils.data.DataLoader(dataTrain,batch_size=32,shuffle=True)

testLoader=torch.utils.data.DataLoader(dataTest,batch_size=32)
def imshow(image, ax=None, title=None, normalize=True):

    """Imshow for Tensor."""

    if ax is None:

        fig, ax = plt.subplots()

    image = image.numpy().transpose((1, 2, 0))



    if normalize:

        mean = np.array([0.485, 0.456, 0.406])

        std = np.array([0.229, 0.224, 0.225])

        image = std * image + mean

        image = np.clip(image, 0, 1)



    ax.imshow(image)

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_visible(False)

    ax.spines['bottom'].set_visible(False)

    ax.tick_params(axis='both', length=0)

    ax.set_xticklabels('')

    ax.set_yticklabels('')



    return ax
images,labels=next(iter(trainLoader))

num=20

img=images[num]

lbl=labels[num]

print('This is {}'.format(lbl))

imshow(img)
model=models.resnet50(pretrained=True)

model.cuda()

model
from collections import OrderedDict

classifier=nn.Sequential(OrderedDict([

    ('fc1',nn.Linear(2048,1024)),

    ('relu1',nn.ReLU()),

    ('drop1',nn.Dropout(p=0.4)),

    ('fc2',nn.Linear(1024,256)),

    ('relu2',nn.ReLU()),

    ('drop2',nn.Dropout(p=0.4)),

    ('fc3',nn.Linear(256,128)),

    ('relu3',nn.ReLU()),

    ('drop3',nn.Dropout(p=0.4)),

    ('fc4',nn.Linear(128,102)),

    ('out',nn.LogSoftmax(dim=1))

]))

for param in model.parameters():

    param.requires_grad=False

model.fc=classifier

model
import json

file=json.load(open('../input/hackathon-blossom-flower-classification/cat_to_name.json'))

NewDict={k: v for k, v in file.items()}

NewDict
classes=list()

for k,v in file.items():

    classes.append(v)

    

classes
criterion=nn.NLLLoss()
model.cuda()
lr=[0.003]

minLoss=1000000

for i in lr:

    model=models.resnet50(pretrained=True)

    classifier=nn.Sequential(OrderedDict([

        ('fc1',nn.Linear(2048,1024)),

        ('relu1',nn.ReLU()),

        ('drop1',nn.Dropout(p=0.4)),

        ('fc2',nn.Linear(1024,256)),

        ('relu2',nn.ReLU()),

        ('drop2',nn.Dropout(p=0.4)),

        ('fc3',nn.Linear(256,128)),

        ('relu3',nn.ReLU()),

        ('drop3',nn.Dropout(p=0.4)),

        ('fc4',nn.Linear(128,102)),

        ('out',nn.LogSoftmax(dim=1))

    ]))

    for param in model.parameters():

        param.requires_grad=False

    model.fc=classifier

    model.cuda()

    optimizer=optim.Adam(model.parameters(),i)

    epochs=30

    print("*********************************************************")

    print("For lr = {}".format(i))

    for j in range(epochs+1):

        trainLoss=0.0

        validLoss=0.0

        for images,labels in trainLoader:

            model.train()

            images,labels=images.cuda(),labels.cuda()

            optimizer.zero_grad()

            output=model(images)

            loss=criterion(output,labels)

            trainLoss+=loss.item()

            loss.backward()

            optimizer.step()

            

        trainLoss=trainLoss/len(trainLoader)

        for imageValid,labelValid in testLoader:

            model.eval()

            imageValid,labelValid=imageValid.cuda(),labelValid.cuda()

            with torch.no_grad():

                log_ps=model(imageValid)

                ValidLoss=criterion(log_ps,labelValid)

                validLoss+=ValidLoss.item()

                

        validLoss=validLoss/len(testLoader)

        if j%2==0:

            print("***************************************************")

            print('Training Loss= {:0.4f} \t Validation Loss= {:0.4f}'.format(trainLoss,validLoss))

                

        if validLoss<minLoss:

            print("************Saving Model**********************")

            torch.save(model.state_dict(),'checkpoint.pth')

            bestlr=i

            minLoss=validLoss
test_loss = 0

with torch.no_grad():

    for images,labels in testLoader:

        images,labels=images.cuda(),labels.cuda()

        output = model(images)

        test_loss += criterion(output,labels).item()

        pred = output.argmax(1, keepdim=True)

    test_loss /= len(testLoader.dataset)



    print('\nAverage loss: {:.4f}'.format(test_loss))
def load_checkpoint(filepath, inference = False):

    checkpoint = torch.load(filepath + 'checkpoint.pth')

    model = checkpoint['model']

    if inference:

        for parameter in model.parameter():

            parameter.require_grad = False

        model.eval()

    model.to(device)

    return model




from PIL import Image



files = os.listdir("../input/hackathon-blossom-flower-classification/test set/test set/")



prediction_list = []



for file in files:

    fullpath = "../input/hackathon-blossom-flower-classification/test set/test set/" + str(file)

    with Image.open(fullpath) as f:

        try:

            img = test_transform(f)

            img = img.unsqueeze(0)

            with torch.no_grad():

                img = img.to(device)

                out = model(img)

                output = torch.exp(out)

                probs,top_class = output.topk(1, dim=1)

                class_name = classes[(top_class.item() - 1)]

                prediction_list.append([file, class_name[1], class_name[0]])

                

        except:

            None

            

df = pd.DataFrame(prediction_list, columns=['Image', 'Flower Name', 'Class Number'])

pd.set_option('display.max_colwidth', -1)

df


