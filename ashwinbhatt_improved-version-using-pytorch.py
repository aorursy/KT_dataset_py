from __future__ import print_function, division

%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import torch

import glob

import re

import torch.nn as nn

import torch.optim as optim

import torchvision

import os

import copy

import cv2 

import torch.nn.functional as F



from PIL import Image

from os import listdir, makedirs, getcwd, remove

from os.path import isfile, join, abspath, exists, isdir, expanduser

from torch.optim import lr_scheduler

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils

from torchvision import datasets, models, transforms

from torch.autograd import Variable

from torch.utils import data

from IPython.display import clear_output



# Ignore warnings

import warnings

warnings.filterwarnings("ignore")



plt.ion()   



import os

print(os.listdir("../input"))
train = glob.glob("../input/dataset/train/*.jpeg")

test = glob.glob("../input/dataset/test/*.jpeg")



from random import shuffle

shuffle(train)



trainset_size = 75000

cross_val = train[trainset_size:]

train = train[:trainset_size]



# CUDA for PyTorch

use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")

torch.backends.cudnn.benchmark = True

# Device configuration
def blockshaped(arr, nrows, ncols):

    h, w = arr.shape

    return (arr.reshape(h//nrows, nrows, -1, ncols)

               .swapaxes(1,2)

               .reshape(-1, nrows, ncols))



def fen_from_filename(filename):

  return os.path.splitext(os.path.basename(filename))[0]



def get_all_labels(list_filename):

    labels = []

    for i in range(len(list_filename)):

        labels.append(fen_from_filename(list_filename[i]))

    return labels
def img_processing(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_shrink = cv2.resize(img_gray, (200, 200))

    box_list = blockshaped(img_shrink, 25, 25)

    flatten_list = box_list.reshape(box_list.shape[0], -1)

    return flatten_list



def fen_to_piece_label(fen):

    y = []

    for i in fen:

        if(str.isdigit(i)):

            d = int(i, 10)

            y.extend(np.zeros(d, np.int16).tolist())

        elif(str.isalpha(i)):

            case = 0

            if(str.isupper(i)):

                case = 6

                i = str.lower(i)

                

            if(i == 'k'):

                case = case + 1

            elif(i == 'q'):

                case = case + 2               

            elif(i == 'r'):

                case = case + 3

            elif(i == 'n'):

                case = case + 4

            elif(i == 'b'):

                case = case + 5

            elif(i == 'p'):

                case = case + 6

            y.append(case)

    return y





def lb_to_fen(label):

    s = ''

    count = 0

    for i in range(len(label)):

        if(i%8 == 0):

            if(count !=0):

                s = s + str(count)

                count =0

            s = s + '-'

        if(label[i]==0):

            count = count+1

        else:

            if(count !=0):

                s = s + str(count)

                count =0

            if(label[i] == 1):

                s = s + 'k'

            elif(label[i] == 2):

                s = s + 'q'

            elif(label[i] == 3):

                s = s + 'r'

            elif(label[i] == 4):

                s = s + 'n'

            elif(label[i] == 5):

                s = s + 'b'

            elif(label[i] == 6):

                s = s + 'p'

            elif(label[i] == 7):

                s = s + 'K'

            elif(label[i] == 8):

                s = s + 'Q'

            elif(label[i] == 9):

                s = s + 'R'

            elif(label[i] == 10):

                s = s + 'N'

            elif(label[i] == 11):

                s = s + 'B'

            elif(label[i] == 12):

                s = s + 'P'

            else:

                print('Invalid Error#######################################')

    if(count != 0):

        s = s+ str(count)

    return s[1:]



def lbs_to_fen(label):

    fen = []

    for grp_no in range(0, int(len(label)/ 64)):

        start_index = grp_no*64

        fen.append(lb_to_fen(label[start_index : start_index +64]))

    return fen   



def decoded_board_with_label(board):

    X = []

    Y = []

    X.extend(img_processing(cv2.imread(board)))

    Y.extend(fen_to_piece_label(fen_from_filename(board)))

    return X, Y





def  get_batch(dataset, batch_size, start_index):

    x = []

    y = []

    for j in range(0, batch_size):

        temp_x, temp_y = decoded_board_with_label(dataset[start_index+ j])

        x.extend(temp_x)

        y.extend(temp_y)

    x = torch.FloatTensor(x)

    y = torch.FloatTensor(y)

    return x, y


###### giving metrics to neural net #####

n_in, n_out = 625, 13

h1 = 60

h2 = 50

h3 = 40

h4 = 30

h5 = 20

batch_size = 5

learning_rate = 1e-4

momentum = 0.9

weight_decay = 0.0000001

epochs  = 5

###############################################






class NeuralNet(nn.Module):

    def __init__(self, n_in, h1, h2, h3, h4, h5, n_out):

        super(NeuralNet, self).__init__()

        self.f1 = nn.Linear(n_in, h1) 

        self.f2 = nn.Linear(h1, h2) 

        self.f3 = nn.Linear(h2, h3)

        self.f4 = nn.Linear(h3, h4)

        self.f5 = nn.Linear(h4, h5)

        self.f6 = nn.Linear(h5, n_out)

    

    def forward(self, x):

        x = F.relu(self.f1(x))

        x = F.relu(self.f2(x))

        x = F.relu(self.f3(x))

        x = F.relu(self.f4(x))

        x = F.relu(self.f5(x))

        x = self.f6(x)        

        return F.log_softmax(x)



# CREATING THE NEURAL NETWORK

model = NeuralNet(n_in, h1, h2, h3, h4, h5, n_out).to(device)



# Loss and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,weight_decay= weight_decay)



for epoch in range(epochs):

    batch_count = int(len(train)/ batch_size)

    for i in range(0, batch_count):  

        

        x_batch, y_batch = get_batch(train, batch_size, batch_size*i)

        

        # Loading batch to the GPU

        x_batch = x_batch.to(device)

        y_batch = y_batch.type(torch.LongTensor).to(device)

        

        # Forward pass

        outputs = model(x_batch)

        loss = criterion(outputs, y_batch)

        if(i%100 ==0):

            print('epoch: ', epoch, 'group no : ', i,'/',batch_count,' loss: ', loss.data.mean())

        # Backward and optimize

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        
def dataset_accuracy_finder(dataset, batch_size):



    # Getting all the actual Test label 

    actual_FEN = get_all_labels(dataset)



    # To store the precdicted labels

    

    Correct_lbs = 0

    y_actual = []

    y_pred = []

    batch_count = int(len(dataset)/ batch_size)

    

    for i in range(0, batch_count):

            x_batch, y_batch = get_batch(dataset, batch_size, batch_size*i)

            y_actual.extend(y_batch.numpy().tolist())

            

            # Loading Features to the GPU

            x_batch = x_batch.to(device)

        

            # Predicting on the model

            y_batch_pred = model(x_batch)

            y_batch_pred = y_batch_pred.view(len(y_batch_pred), -1).argmax(1).cpu().numpy().astype(np.int32).tolist()

            y_pred.extend(y_batch_pred)



    y_actual = np.array(y_actual)

    y_pred = np.array(y_pred)

    Correct_lbs = np.count_nonzero(y_actual == y_pred)

    

    

    pred_FEN = lbs_to_fen(y_pred)

    

    # Calculating the real Test Set Accuracy

    Correct_fen = 0

    for i in range(0, len(actual_FEN)):

        if(pred_FEN[i] == actual_FEN[i]):

            Correct_fen = Correct_fen + 1

    return Correct_lbs/(len(y_pred)), Correct_fen/len(actual_FEN),  

train_err, train_actual_err = dataset_accuracy_finder(train, 500)

cross_val_err, cross_val_actual_err = dataset_accuracy_finder(cross_val, 500)

test_err, test_actual_err = dataset_accuracy_finder(test, 500)
print('============================================================================================')

print('    Dataset                   (  Label   )                  (  FEN code  )    ')

print('    Train                      ',train_err, '                ' , train_actual_err)

print('    DEV                        ',cross_val_err, '              ' , cross_val_actual_err)

print('    Test                       ',test_err, '                   ' , test_actual_err)
