# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

import cv2

from glob import glob

import torchvision

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

import torchvision.models as models

from torch.autograd import Variable

import torchvision.transforms as transforms

from PIL import Image, ImageFile

from torch.utils.data import TensorDataset, DataLoader,Dataset

from sklearn.model_selection import train_test_split

import copy 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

#print(os.listdir("../input"))

#../input/train/train

#../input/train/test



# Any results you write to the current directory are saved as output.

use_cuda = torch.cuda.is_available()

if not use_cuda:

    print('No GPU found. Please use a GPU to train your neural network.')
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/sample_submission.csv')
#df_train = df_train[:5000]

print(df_train.shape)

df_train.head()
# NOTE: class is inherited from Dataset

class ImageLabelDataset(Dataset):

    def __init__(self, df_data, prediction, folder="../input/train_images"):

        super().__init__()

        self.df = df_data.values

        self.prediction = prediction.values

        self.folder = folder



    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, index):

        tensorimage = self.preprocess_image(self.df[index])

        label = self.prediction[index]

        label_tensor = self.get_dummies(label)

        #label_tensor = label

        #x = np.squeeze(label_tensor.detach().cpu().numpy())

        #print(x.argsort()[-6:][::-1])

        return tensorimage, label_tensor

    

    def preprocess_image(self, img_path):

        data_transform = transforms.Compose([transforms.ToPILImage(),

                                             transforms.Resize(299), 

                                             transforms.RandomResizedCrop(299), 

                                             transforms.RandomHorizontalFlip(),

                                             transforms.RandomVerticalFlip(),

                                             transforms.RandomRotation(degrees=30), 

                                             transforms.ToTensor(),

                                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

                                            ])

        image = cv2.imread("{}/{}.png".format(self.folder, img_path))

        image = data_transform(image)

        return image

    

    def get_dummies(self, attribute_id):

        label_tensor = torch.zeros((1, 5))

        #for label in attribute_id.split():

        label_tensor[0, int(attribute_id)] = 1

        return label_tensor
# df_train.head()

batch_size = 8

train_image = df_train["id_code"]

#target = df_train.drop(['id', 'attribute_ids'],axis=1)

target = df_train["diagnosis"]

X_train, X_val, y_train, y_val = train_test_split(train_image, target, random_state=42, test_size=0.1)



test_image = df_test["id_code"]

test_target = df_test["diagnosis"]

#test_target = df_test.drop(['id', 'attribute_ids'],axis=1)



train_set = ImageLabelDataset(df_data=X_train, prediction=y_train, folder="../input/train_images")

val_set = ImageLabelDataset(df_data=X_val, prediction=y_val, folder="../input/train_images")

predict_set = ImageLabelDataset(df_data=test_image, prediction=test_target, folder="../input/test_images")



train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)

val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=1)

test_val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)

test_loader = torch.utils.data.DataLoader(predict_set, batch_size=1, num_workers=0)

# Hyperparameters

n_output=1

# Number of Epochs

num_epochs = 4

# Learning Rate

learning_rate = 0.001

# Model parameters



# Show stats for every n number of batches

show_every_n_batches = 1
def train_rnn(model, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):

    train_loss = 0

    valid_loss = 0

    batch_losses = []

    val_batch_losses = []

    valid_loss_min = np.Inf

    

    model.train()

    

    previousLoss = np.Inf

    minLoss = np.Inf



    print("Training for %d epoch(s)..." % n_epochs)

    for epoch_i in range(1, n_epochs + 1):

        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):

            

            if use_cuda:

                data, target = data.cuda(), target.cuda()

            # clear the gradients of all optimized variables

            target = np.squeeze(target)

            #target = target.view(1,-1)

            target= target.long()

            #print(np.squeeze(target))

            #targs = target.view(-1).long()

            # print(np.squeeze(target))

            #print(torch.max(target, 1)[1])

            #print(data)

            optimizer.zero_grad()

            # forward pass: Train compute predicted outputs by passing inputs to the model

            output = model.forward(data)

            output, _ = output

            #print(output.squeeze(1))

            #print(type(output))

            #_, preds = torch.max(output.data, 1)

            #print(torch.max(target, 1)[1])

            #print("output")

            #print(torch.max(output, 1)[1])

            # calculate the Train batch loss

            #output = Variable(output.float())

            #target = Variable(target.long())

            loss = criterion(output, torch.max(target, 1)[1])

            # backward pass: compute gradient of the loss with respect to model parameters

            loss.backward()

            # perform a single optimization step (parameter update)

            optimizer.step()

            # update training loss

            ## find the loss and update the model parameters accordingly

            ## record the average training loss, using something like

            #train_loss += ((1 / (batch_idx + 1)) * (loss.item() - train_loss))

            train_loss += loss.item()

            

        model.eval()

        for batch_idx, (data, target) in enumerate(val_loader):

            # move to GPU

            if use_cuda:

                data, target = data.cuda(), target.cuda()

            ## update the average validation loss

            # forward pass: compute predicted outputs by passing inputs to the model

            target = np.squeeze(target)

            target= target.long()

            #target = target.view(1,-1)

            

            output = model.forward(data)

            #print(output)

            #output, _ = output

            # calculate the Val batch loss

            loss = criterion(output, torch.max(target, 1)[1])

            # update average validation loss 

            #val_batch_losses.append(loss.item())

            #valid_loss += ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))

            valid_loss += loss.item()

        

        valid_loss = valid_loss/len(val_loader.dataset)

        train_loss = train_loss/len(train_loader.dataset)

        

        # print training/validation statistics 

        if epoch_i%show_every_n_batches == 0:

            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch_i, train_loss, valid_loss))

        

            ## TODO: save the model if validation loss has decreased

            # save model if validation loss has decreased

            if valid_loss < valid_loss_min:

                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

                valid_loss_min,

                valid_loss))

                with open('trained_rnn_new', 'wb') as pickle_file:

                    torch.save(model.state_dict(), 'trained_rnn_new')

                valid_loss_min = valid_loss

                train_loss = 0

                valid_loss = 0

                #batch_losses = []

                #val_batch_losses = []

  

    return model
model_transfer = models.inception_v3(pretrained=True)

print(model_transfer)
# Freeze training for all "features" layers

#for param in model_transfer.features.parameters():

    #param.requires_grad = False



## Freezing all layers

for params in model_transfer.parameters():

    params.requires_grad = False



## Freezing the first few layers. Here I am freezing the first 7 layers 

        

custom_model = nn.Sequential(nn.Linear(2048, 1024), 

                  nn.ReLU(),

                  nn.Dropout(p=0.5), 

                  nn.Linear(1024, 5)

                 )



model_transfer.fc = custom_model

#model_transfer.classifier = custom_model



if use_cuda:

    model_transfer = model_transfer.cuda()



# print(model_transfer)



# specify loss function

criterion_scratch = nn.CrossEntropyLoss()

#criterion_scratch = nn.BCELoss()

#criterion_scratch = nn.BCELoss(reduction="mean").to('cuda:0')



# specify optimizer

optimizer_scratch = optim.SGD(model_transfer.parameters(), lr=learning_rate, momentum=0.9)

#optimizer_scratch = optim.Adam(model_transfer.classifier.parameters(), lr=learning_rate)

#optimizer_scratch = optim.SGD(list(filter(lambda p: p.requires_grad, model_transfer.parameters())), lr=learning_rate, momentum=0.9)





#trained_rnn = train_rnn(model_transfer, batch_size, optimizer_scratch, criterion_scratch, num_epochs, show_every_n_batches)
model_transfer.load_state_dict(torch.load('trained_rnn_new'))

model_transfer.eval()
from sklearn.metrics import r2_score



preds = []

for batch_idx, (data, target) in enumerate(test_val_loader):

    # move to GPU

    #print(np.squeeze(target.detach().cpu().numpy()))

    if use_cuda:

        data, target = data.cuda(), target.cuda()

    ## update the average validation loss

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model_transfer.forward(data)

    #print(target)

    #print(output)

    #print(torch.max(output, 1)[1].detach().cpu().numpy())

    #values, indices = output.max(1)    

    pr = torch.max(output, 1)[1].detach().cpu().numpy()

    preds.append(pr[0])

    #print(pr.detach().cpu())

print(preds[:100])

r2_score(y_val.values, preds)
preds = []

np.set_printoptions(threshold=100)

submission = pd.read_csv('../input/sample_submission.csv')

for batch_idx, (data, target) in enumerate(test_loader):

    # move to GPU

    #print(np.squeeze(target.detach().cpu().numpy()))

    if use_cuda:

        data, target = data.cuda(), target.cuda()

    ## update the average validation loss

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model_transfer.forward(data)

    #print(target)

    #print(output)

    #print(torch.max(output, 1)[1].detach().cpu().numpy())

    #values, indices = output.max(1)    

    pr = torch.max(output, 1)[1].detach().cpu().numpy()

    preds.append(pr[0])

    #for i in pr:

        #preds.append(i)

    #print(batch_idx)

    #print(pr)

    #print(x.argsort()[-6:][::-1])

    #print(pr.argsort()[-6:][::-1])

    #print(values)

    # calculate the batch loss

submission["diagnosis"] = preds

print(submission.head())

#submission.to_csv('submission.csv', index=False)
# import the modules we'll need

from IPython.display import HTML

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename="submission_1.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe

create_download_link(submission)
predictoutput = pd.read_csv("../input/mydatasetforaerialcactus/submission_2.csv")

predictoutput.to_csv('submission.csv', index=False)