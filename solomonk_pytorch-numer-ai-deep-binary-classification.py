

# !pip install http://download.pytorch.org/whl/cu75/torch-0.2.0.post1-cp27-cp27mu-manylinux1_x86_64.whl

# !pip install --upgrade torch

# !pip install torchvision 

# ! pip install cv2

# import cv2



# !pip install pycuda

%reset -f

# %%timeit



import torch

from torch.autograd import Variable

import numpy as np

import pandas

import numpy as np

import pandas as pd

from sklearn import cross_validation

from sklearn import metrics

from sklearn.metrics import roc_auc_score, log_loss, roc_auc_score, roc_curve, auc

import matplotlib.pyplot as plt

from sklearn import cross_validation

from sklearn import metrics

from sklearn.metrics import roc_auc_score, log_loss, roc_auc_score, roc_curve, auc

from sklearn.cross_validation import StratifiedKFold, ShuffleSplit, cross_val_score, train_test_split

import logging

import numpy

import numpy as np

from __future__ import print_function

from __future__ import division

import math

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

import os

import torch

from torch.utils.data.dataset import Dataset

from torch.utils.data import DataLoader

from torchvision import transforms

from torch import nn

import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable

from sklearn.preprocessing import MultiLabelBinarizer

import time

from sklearn.preprocessing import PolynomialFeatures

import pandas as pd

import numpy as np

import scipy

%matplotlib inline

from pylab import rcParams

rcParams['figure.figsize'] = (6, 6)      # setting default size of plots

import tensorflow as tf 

print("tensorflow:" + tf.__version__)

!set "KERAS_BACKEND=tensorflow"

import torch

import sys

print('__Python VERSION:', sys.version)

print('__pyTorch VERSION:', torch.__version__)

print('__CUDA VERSION')

from subprocess import call

print('__CUDNN VERSION:', torch.backends.cudnn.version())

print('__Number CUDA Devices:', torch.cuda.device_count())

print('__Devices')





print("OS: ", sys.platform)

print("Python: ", sys.version)

print("PyTorch: ", torch.__version__)

print("Numpy: ", np.__version__)



handler=logging.basicConfig(level=logging.INFO)

lgr = logging.getLogger(__name__)

%matplotlib inline



# !pip install psutil

import psutil

def cpuStats():

        print(sys.version)

        print(psutil.cpu_percent())

        print(psutil.virtual_memory())  # physical memory usage

        pid = os.getpid()

        py = psutil.Process(pid)

        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think

        print('memory GB:', memoryUse)



cpuStats()
# %%timeit

use_cuda = torch.cuda.is_available()

# use_cuda = False



FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

Tensor = FloatTensor



lgr.info("USE CUDA=" + str (use_cuda))



# ! watch -n 0.1 'ps f -o user,pgrp,pid,pcpu,pmem,start,time,command -p `lsof -n -w -t /dev/nvidia*`'

# sudo apt-get install dstat #install dstat

# sudo pip install nvidia-ml-py #install Python NVIDIA Management Library

# wget https://raw.githubusercontent.com/datumbox/dstat/master/plugins/dstat_nvidia_gpu.py

# sudo mv dstat_nvidia_gpu.py /usr/share/dstat/ #move file to the plugins directory of dstat
# Data params

TARGET_VAR= 'target'

TOURNAMENT_DATA_CSV = 'numerai_tournament_data.csv'

TRAINING_DATA_CSV = 'numerai_training_data.csv'

BASE_FOLDER = '../input/'



# fix seed

seed=17*19

np.random.seed(seed)

torch.manual_seed(seed)

if use_cuda:

    torch.cuda.manual_seed(seed)
# %%timeit

df_train = pd.read_csv(BASE_FOLDER + TRAINING_DATA_CSV)

df_train.head(5)
from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

from collections import defaultdict



# def genBasicFeatures(inDF):

#     print('Generating basic features ...')

#     df_copy=inDF.copy(deep=True)

#     magicNumber=21

#     feature_cols = list(inDF.columns)



#     inDF['x_mean'] = np.mean(df_copy.ix[:, 0:magicNumber], axis=1)

#     inDF['x_median'] = np.median(df_copy.ix[:, 0:magicNumber], axis=1)

#     inDF['x_std'] = np.std(df_copy.ix[:, 0:magicNumber], axis=1)

#     inDF['x_skew'] = scipy.stats.skew(df_copy.ix[:, 0:magicNumber], axis=1)

#     inDF['x_kurt'] = scipy.stats.kurtosis(df_copy.ix[:, 0:magicNumber], axis=1)

#     inDF['x_var'] = np.var(df_copy.ix[:, 0:magicNumber], axis=1)

#     inDF['x_max'] = np.max(df_copy.ix[:, 0:magicNumber], axis=1)

#     inDF['x_min'] = np.min(df_copy.ix[:, 0:magicNumber], axis=1)    



#     return inDF



def addPolyFeatures(inDF, deg=2):

    print('Generating poly features ...')

    df_copy=inDF.copy(deep=True)

    poly=PolynomialFeatures(degree=deg)

    p_testX = poly.fit(df_copy)

    # AttributeError: 'PolynomialFeatures' object has no attribute 'get_feature_names'

    target_feature_names = ['x'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(df_copy.columns,p) for p in poly.powers_]]

    df_copy = pd.DataFrame(p_testX.transform(df_copy),columns=target_feature_names)

        

    return df_copy



def oneHOT(inDF):

    d = defaultdict(LabelEncoder)

    X_df=inDF.copy(deep=True)

    # Encoding the variable

    X_df = X_df.apply(lambda x: d['era'].fit_transform(x))

            

    return X_df

    
from sklearn import preprocessing



# Train, Validation, Test Split

def loadDataSplit():

    df_train = pd.read_csv(BASE_FOLDER + TRAINING_DATA_CSV)

    # TOURNAMENT_DATA_CSV has both validation and test data provided by NumerAI

    df_test_valid = pd.read_csv(BASE_FOLDER + TOURNAMENT_DATA_CSV)



    answers_1_SINGLE = df_train[TARGET_VAR]

    df_train.drop(TARGET_VAR, axis=1,inplace=True)

    df_train.drop('id', axis=1,inplace=True)

    df_train.drop('era', axis=1,inplace=True)

    df_train.drop('data_type', axis=1,inplace=True)    

    

#     df_train=oneHOT(df_train)



    df_train.to_csv(TRAINING_DATA_CSV + 'clean.csv', header=False,  index = False)    

    df_train= pd.read_csv(TRAINING_DATA_CSV + 'clean.csv', header=None, dtype=np.float32)    

    df_train = pd.concat([df_train, answers_1_SINGLE], axis=1)

    feature_cols = list(df_train.columns[:-1])

#     print (feature_cols)

    target_col = df_train.columns[-1]

    trainX, trainY = df_train[feature_cols], df_train[target_col]

    

    

    # TOURNAMENT_DATA_CSV has both validation and test data provided by NumerAI

    # Validation set

    df_validation_set=df_test_valid.loc[df_test_valid['data_type'] == 'validation'] 

    df_validation_set=df_validation_set.copy(deep=True)

    answers_1_SINGLE_validation = df_validation_set[TARGET_VAR]

    df_validation_set.drop(TARGET_VAR, axis=1,inplace=True)    

    df_validation_set.drop('id', axis=1,inplace=True)

    df_validation_set.drop('era', axis=1,inplace=True)

    df_validation_set.drop('data_type', axis=1,inplace=True)

    

#     df_validation_set=oneHOT(df_validation_set)

    

    df_validation_set.to_csv(TRAINING_DATA_CSV + '-validation-clean.csv', header=False,  index = False)    

    df_validation_set= pd.read_csv(TRAINING_DATA_CSV + '-validation-clean.csv', header=None, dtype=np.float32)    

    df_validation_set = pd.concat([df_validation_set, answers_1_SINGLE_validation], axis=1)

    feature_cols = list(df_validation_set.columns[:-1])



    target_col = df_validation_set.columns[-1]

    valX, valY = df_validation_set[feature_cols], df_validation_set[target_col]

                            

    # Test set for submission (not labeled)    

    df_test_set = pd.read_csv(BASE_FOLDER + TOURNAMENT_DATA_CSV)

#     df_test_set=df_test_set.loc[df_test_valid['data_type'] == 'live'] 

    df_test_set=df_test_set.copy(deep=True)

    df_test_set.drop(TARGET_VAR, axis=1,inplace=True)

    tid_1_SINGLE = df_test_set['id']

    df_test_set.drop('id', axis=1,inplace=True)

    df_test_set.drop('era', axis=1,inplace=True)

    df_test_set.drop('data_type', axis=1,inplace=True)   

    

#     df_test_set=oneHOT(df_validation_set)

    

    feature_cols = list(df_test_set.columns) # must be run here, we dont want the ID    

#     print (feature_cols)

    df_test_set = pd.concat([tid_1_SINGLE, df_test_set], axis=1)            

    testX = df_test_set[feature_cols].values

        

    return trainX, trainY, valX, valY, testX, df_test_set
# %%timeit

trainX, trainY, valX, valY, testX, df_test_set = loadDataSplit()



min_max_scaler = preprocessing.MinMaxScaler()

    

# # Number of features for the input layer

N_FEATURES=trainX.shape[1]

print (trainX.shape)

print (trainY.shape)

print (valX.shape)

print (valY.shape)

print (testX.shape)

print (df_test_set.shape)



# print (trainX)
# seperate out the Categorical and Numerical features

import seaborn as sns



numerical_feature=trainX.dtypes[trainX.dtypes!= 'object'].index

categorical_feature=trainX.dtypes[trainX.dtypes== 'object'].index



print ("There are {} numeric and {} categorical columns in train data".format(numerical_feature.shape[0],categorical_feature.shape[0]))



corr=trainX[numerical_feature].corr()

sns.heatmap(corr)

from pandas import *

import numpy as np

from scipy.stats.stats import pearsonr

import itertools



# from https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas

def get_redundant_pairs(df):

    '''Get diagonal and lower triangular pairs of correlation matrix'''

    pairs_to_drop = set()

    cols = df.columns

    for i in range(0, df.shape[1]):

        for j in range(0, i+1):

            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop



def get_top_abs_correlations(df, n=5):

    au_corr = df.corr().abs().unstack()

    labels_to_drop = get_redundant_pairs(df)

    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    return au_corr[0:n]



print("Top Absolute Correlations")

print(get_top_abs_correlations(trainX, 5))
# Convert the np arrays into the correct dimention and type

# Note that BCEloss requires Float in X as well as in y

def XnumpyToTensor(x_data_np):

    x_data_np = np.array(x_data_np, dtype=np.float32)        

    print(x_data_np.shape)

    print(type(x_data_np))



    if use_cuda:

        lgr.info ("Using the GPU")    

        X_tensor = Variable(torch.from_numpy(x_data_np).cuda()) # Note the conversion for pytorch    

    else:

        lgr.info ("Using the CPU")

        X_tensor = Variable(torch.from_numpy(x_data_np)) # Note the conversion for pytorch

    

    print(type(X_tensor.data)) # should be 'torch.cuda.FloatTensor'

    print(x_data_np.shape)

    print(type(x_data_np))    

    return X_tensor





# Convert the np arrays into the correct dimention and type

# Note that BCEloss requires Float in X as well as in y

def YnumpyToTensor(y_data_np):    

    y_data_np=y_data_np.reshape((y_data_np.shape[0],1)) # Must be reshaped for PyTorch!

    print(y_data_np.shape)

    print(type(y_data_np))



    if use_cuda:

        lgr.info ("Using the GPU")            

    #     Y = Variable(torch.from_numpy(y_data_np).type(torch.LongTensor).cuda())

        Y_tensor = Variable(torch.from_numpy(y_data_np)).type(torch.FloatTensor).cuda()  # BCEloss requires Float        

    else:

        lgr.info ("Using the CPU")        

    #     Y = Variable(torch.squeeze (torch.from_numpy(y_data_np).type(torch.LongTensor)))  #         

        Y_tensor = Variable(torch.from_numpy(y_data_np)).type(torch.FloatTensor)  # BCEloss requires Float        



    print(type(Y_tensor.data)) # should be 'torch.cuda.FloatTensor'

    print(y_data_np.shape)

    print(type(y_data_np))    

    return Y_tensor
# p is the probability of being dropped in PyTorch



# NN params

DROPOUT_PROB = 0.90



LR = 0.005

MOMENTUM= 0.9

dropout = torch.nn.Dropout(p=1 - (DROPOUT_PROB))



lgr.info(dropout)



hiddenLayer1Size=512

hiddenLayer2Size=int(hiddenLayer1Size/4)

hiddenLayer3Size=int(hiddenLayer1Size/8)

hiddenLayer4Size=int(hiddenLayer1Size/16)

hiddenLayer5Size=int(hiddenLayer1Size/32)



linear1=torch.nn.Linear(N_FEATURES, hiddenLayer1Size, bias=True) 

torch.nn.init.xavier_uniform(linear1.weight)



linear2=torch.nn.Linear(hiddenLayer1Size, hiddenLayer2Size)

torch.nn.init.xavier_uniform(linear2.weight)



linear6=torch.nn.Linear(hiddenLayer2Size, 1)

torch.nn.init.xavier_uniform(linear6.weight)



sigmoid = torch.nn.Sigmoid()

tanh=torch.nn.Tanh()

relu=torch.nn.LeakyReLU()



net = torch.nn.Sequential(linear1,nn.BatchNorm1d(hiddenLayer1Size),relu,

                          linear2,dropout,relu,

                          linear6,sigmoid

                          )

lgr.info(net)  # net architecture
# optimizer = torch.optim.SGD(net.parameters(), lr=0.02)

# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=5e-3)

#L2 regularization can easily be added to the entire model via the optimizer

optimizer = torch.optim.Adam(net.parameters(), lr=LR,weight_decay=5e-3) #  L2 regularization



loss_func=torch.nn.BCELoss() # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss

# http://andersonjo.github.io/artificial-intelligence/2017/01/07/Cost-Functions/



if use_cuda:

    lgr.info ("Using the GPU")    

    net.cuda()

    loss_func.cuda()

#     cudnn.benchmark = True



lgr.info (optimizer)

lgr.info (loss_func)
import time

start_time = time.time()    

epochs=60 # change to 1500 for better results

all_losses = []



X_tensor_train= XnumpyToTensor(trainX)

Y_tensor_train= YnumpyToTensor(trainY)



print(type(X_tensor_train.data), type(Y_tensor_train.data)) # should be 'torch.cuda.FloatTensor'



# From here onwards, we must only use PyTorch Tensors

for step in range(epochs):    

    out = net(X_tensor_train)                 # input x and predict based on x

    cost = loss_func(out, Y_tensor_train)     # must be (1. nn output, 2. target), the target label is NOT one-hotted



    optimizer.zero_grad()   # clear gradients for next train

    cost.backward()         # backpropagation, compute gradients

    optimizer.step()        # apply gradients

                   

        

    if step % 5 == 0:        

        loss = cost.data[0]

        all_losses.append(loss)

        print(step, cost.data.cpu().numpy())

        # RuntimeError: can't convert CUDA tensor to numpy (it doesn't support GPU arrays). 

        # Use .cpu() to move the tensor to host memory first.        

        prediction = (net(X_tensor_train).data).float() # probabilities         

#         prediction = (net(X_tensor).data > 0.5).float() # zero or one

#         print ("Pred:" + str (prediction)) # Pred:Variable containing: 0 or 1

#         pred_y = prediction.data.numpy().squeeze()            

        pred_y = prediction.cpu().numpy().squeeze()

        target_y = Y_tensor_train.cpu().data.numpy()

                        

        tu = (log_loss(target_y, pred_y),roc_auc_score(target_y,pred_y ))

        print ('LOG_LOSS={}, ROC_AUC={} '.format(*tu))        

                

end_time = time.time()

print ('{} {:6.3f} seconds'.format('GPU:', end_time-start_time))



%matplotlib inline

import matplotlib.pyplot as plt

plt.plot(all_losses)

plt.show()



false_positive_rate, true_positive_rate, thresholds = roc_curve(target_y,pred_y)

roc_auc = auc(false_positive_rate, true_positive_rate)



plt.title('LOG_LOSS=' + str(log_loss(target_y, pred_y)))

plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.6f' % roc_auc)

plt.legend(loc='lower right')

plt.plot([0, 1], [0, 1], 'r--')

plt.xlim([-0.1, 1.2])

plt.ylim([-0.1, 1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
net.eval()

# Validation data

print (valX.shape)

print (valY.shape)



X_tensor_val= XnumpyToTensor(valX)

Y_tensor_val= YnumpyToTensor(valY)





print(type(X_tensor_val.data), type(Y_tensor_val.data)) # should be 'torch.cuda.FloatTensor'



predicted_val = (net(X_tensor_val).data).float() # probabilities 

# predicted_val = (net(X_tensor_val).data > 0.5).float() # zero or one

pred_y = predicted_val.cpu().numpy()

target_y = Y_tensor_val.cpu().data.numpy()                



print (type(pred_y))

print (type(target_y))



tu = (str ((pred_y == target_y).mean()),log_loss(target_y, pred_y),roc_auc_score(target_y,pred_y ))

print ('\n')

print ('acc={} log_loss={} roc_auc={} '.format(*tu))



false_positive_rate, true_positive_rate, thresholds = roc_curve(target_y,pred_y)

roc_auc = auc(false_positive_rate, true_positive_rate)



plt.title('LOG_LOSS=' + str(log_loss(target_y, pred_y)))

plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.6f' % roc_auc)

plt.legend(loc='lower right')

plt.plot([0, 1], [0, 1], 'r--')

plt.xlim([-0.1, 1.2])

plt.ylim([-0.1, 1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()



# print (pred_y)
# testX, df_test_set

# df[df.columns.difference(['b'])]

# trainX, trainY, valX, valY, testX, df_test_set = loadDataSplit()

    

print (df_test_set.shape)

columns = ['id', 'probability']

df_pred=pd.DataFrame(data=np.zeros((0,len(columns))), columns=columns)

# df_pred.id.astype(int)



for index, row in df_test_set.iterrows():

    rwo_no_id=row.drop('id')    

#     print (rwo_no_id.values)    

    x_data_np = np.array(rwo_no_id.values, dtype=np.float32)        

    if use_cuda:

        X_tensor_test = Variable(torch.from_numpy(x_data_np).cuda()) # Note the conversion for pytorch    

    else:

        X_tensor_test = Variable(torch.from_numpy(x_data_np)) # Note the conversion for pytorch

                    

    X_tensor_test=X_tensor_test.view(1, trainX.shape[1]) # does not work with 1d tensors            

    predicted_val = (net(X_tensor_test).data).float() # probabilities     

    p_test =   predicted_val.cpu().numpy().item() # otherwise we get an array, we need a single float

    

    df_pred = df_pred.append({'id':row['id'], 'probability':p_test},ignore_index=True)

#     df_pred = df_pred.append({'id':row['id'].astype(int), 'probability':p_test},ignore_index=True)



df_pred.head(5)
# df_pred.id=df_pred.id.astype(int)



def savePred(df_pred, loss):

#     csv_path = 'pred/p_{}_{}_{}.csv'.format(loss, name, (str(time.time())))

    csv_path = 'pred/pred_{}_{}.csv'.format(loss, (str(time.time())))

    df_pred.to_csv(csv_path, columns=('id', 'probability'), index=None)

    print (csv_path)

    

savePred (df_pred, log_loss(target_y, pred_y))