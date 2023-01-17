%reset -f

from __future__ import print_function

from __future__ import division





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import torch

import sys

import torch

from torch.utils.data.dataset import Dataset

from torch.utils.data import DataLoader

from torchvision import transforms

from torch import nn

import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable



from sklearn import cross_validation

from sklearn import metrics

from sklearn.metrics import roc_auc_score, log_loss, roc_auc_score, roc_curve, auc

from sklearn.cross_validation import StratifiedKFold, ShuffleSplit, cross_val_score, train_test_split



print('__Python VERSION:', sys.version)

print('__pyTorch VERSION:', torch.__version__)



import numpy

import numpy as np







# ! watch -n 0.1 'ps f -o user,pgrp,pid,pcpu,pmem,start,time,command -p `lsof -n -w -t /dev/nvidia*`'

# sudo apt-get install dstat #install dstat

# sudo pip install nvidia-ml-py #install Python NVIDIA Management Library

# wget https://raw.githubusercontent.com/datumbox/dstat/master/plugins/dstat_nvidia_gpu.py

# sudo mv dstat_nvidia_gpu.py /usr/share/dstat/ #move file to the plugins directory of dstat



import pandas

import pandas as pd



import logging

handler=logging.basicConfig(level=logging.INFO)

lgr = logging.getLogger(__name__)





# !pip install psutil

import psutil

import os

def cpuStats():

        print(sys.version)

        print(psutil.cpu_percent())

        print(psutil.virtual_memory())  # physical memory usage

        pid = os.getpid()

        py = psutil.Process(pid)

        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think

        print('memory GB:', memoryUse)



cpuStats()



# Data params

TARGET_VAR= 'target'

BASE_FOLDER = '../input/'


use_cuda = torch.cuda.is_available()

# use_cuda = False



FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

Tensor = FloatTensor



print("USE CUDA=" + str (use_cuda))



#torch.backends.cudnn.benchmark = True
# fix seed

seed=17*19

np.random.seed(seed)

torch.manual_seed(seed)

if use_cuda:

    torch.cuda.manual_seed(seed)



#####

# Load in the data

#####

print('loading data')



df_train = pd.read_csv('../input/train.tsv', sep='\t')

df_test = pd.read_csv('../input/test.tsv', sep='\t')



print('Train shape:{}\nTest shape:{}'.format(df_train.shape, df_test.shape))



df_train.head(5)
# df_train.plot(kind='scatter', x='item_condition_id', y='price', title='Weight and height in adults')

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

from collections import defaultdict

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from sklearn import preprocessing



d = defaultdict(LabelEncoder)

TARGET_VAR='price'



def split_cat(s):

    try:

        return s.split('/')[0],s.split('/')[1],s.split('/')[2],

    except:

        return [0,0,0]



df_train[['cat1','cat2','cat3']] = pd.DataFrame(df_train.category_name.apply(split_cat).tolist(),

                                   columns = ['cat1','cat2','cat3'])

df_test[['cat1','cat2','cat3']] = pd.DataFrame(df_test.category_name.apply(split_cat).tolist(),

                                   columns = ['cat1','cat2','cat3'])



print('making the magic...')

corpus1 = df_train.name.values.astype('U').tolist() + df_test.name.values.astype('U').tolist()

corpus2 = df_train.item_description.values.astype('U').tolist() + df_test.item_description.values.astype('U').tolist()



vectorizer1 = CountVectorizer(min_df=1,stop_words='english')

vectorizer1.fit(corpus1)



vectorizer2 = CountVectorizer(min_df=1,stop_words='english')

vectorizer2.fit(corpus2)



train_cor1 = vectorizer1.transform(df_train.name.values.astype('U').tolist())

train_cor2 = vectorizer2.transform(df_train.item_description.values.astype('U').tolist())



test_cor1 = vectorizer1.transform(df_test.name.values.astype('U').tolist())

test_cor2 = vectorizer2.transform(df_test.item_description.values.astype('U').tolist())





df_train['cor1'] = np.mean(train_cor1,1)

df_train['cor2'] = np.mean(train_cor2,1)



df_test['cor1'] = np.mean(test_cor1,1)

df_test['cor2'] = np.mean(test_cor2,1)





df_train['len1'] = df_train.name.str.len()

df_train['len2'] = df_train.item_description.str.len()



df_test['len1'] = df_test.name.str.len()

df_test['len2'] = df_test.item_description.str.len()



print("label encoding...")

le = preprocessing.LabelEncoder()

le.fit(df_train.brand_name.values.tolist() + df_test.brand_name.values.tolist())

df_train['brands']= le.transform(df_train.brand_name.values.tolist())

df_test['brands']= le.transform(df_test.brand_name.values.tolist())



df_train = df_train.fillna(999)

df_test = df_test.fillna(999)



le = preprocessing.LabelEncoder()

le.fit(df_train.cat1.values.tolist() + df_test.cat1.values.tolist())

df_train['cat1']= le.transform(df_train.cat1.values.tolist())



le = preprocessing.LabelEncoder()

le.fit(df_train.cat2.values.tolist() + df_test.cat2.values.tolist())

df_train['cat2']= le.transform(df_train.cat2.values.tolist())



le = preprocessing.LabelEncoder()

le.fit(df_train.cat3.values.tolist() + df_test.cat3.values.tolist())

df_train['cat3']= le.transform(df_train.cat3.values.tolist())



df_train = df_train.fillna(999)

df_test = df_test.fillna(999)





answers_1_SINGLE = np.abs(df_train[TARGET_VAR])

drop_features = ['train_id', 'name', 'category_name', 'brand_name', 'price', 'item_description']

df_train = df_train.drop(drop_features, axis=1)



df_train = df_train.fillna(999)

df_test = df_test.fillna(999)



df_train.head()

df_train.to_csv('train_clean.csv', header=False,  index = False)    

df_train= pd.read_csv('train_clean.csv', header=None, dtype=np.float32)    

df_train = pd.concat([df_train, answers_1_SINGLE], axis=1)

feature_cols = list(df_train.columns[:-1])

print (feature_cols)

target_col = df_train.columns[-1]

trainX, trainY = df_train[feature_cols], df_train[target_col]

df_train.head()
# Make sure the shape and data are OK

# Make sure the shape and data are OK

print(trainX.shape)

print(trainY.shape)

print(type(trainY))

print(type(trainY))



from sklearn.model_selection import train_test_split



data_train, data_val, labels_train, labels_val = train_test_split(trainX, trainY, 

                                                                    test_size=0.20, random_state=999)

data_train=data_train.values

labels_train=labels_train.values

print(data_train.shape)

print(labels_train.shape)

print(type(data_train))

print(type(labels_train))



data_val=data_val.values

labels_val=labels_val.values

print(data_val.shape)

print(labels_val.shape)

print(type(data_val))

print(type(labels_val))

print('designing model')

# Training Parameters

learning_rate = 0.005

# Network Parameters

N_FEATURES=data_train.shape[1] # # Number of features for the input layer

num_classes = 1 # Linear

dropout = 0.5 # Dropout, probability to keep units

print ('Num of features:' + str (N_FEATURES))
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
DEBUG_ON=True

def debug(msg, x):

    if DEBUG_ON:

        print (msg + ', (size():' + str (x.size()))



dropout = torch.nn.Dropout(0.3)

relu=torch.nn.LeakyReLU()

N_HIDDEN=16



net_overfitting = torch.nn.Sequential(

    torch.nn.Linear(N_FEATURES, N_HIDDEN),

    torch.nn.ReLU(),

    torch.nn.Linear(N_HIDDEN, N_HIDDEN),

    torch.nn.ReLU(),

    torch.nn.Linear(N_HIDDEN, 1),

)



net_dropped = torch.nn.Sequential(

    torch.nn.Linear(N_FEATURES, N_HIDDEN),

    nn.BatchNorm1d(N_HIDDEN),

    torch.nn.Dropout(0.3),

    torch.nn.ReLU(),

    torch.nn.Linear(N_HIDDEN, N_HIDDEN),

    torch.nn.Dropout(0.3),

    torch.nn.ReLU(),

    torch.nn.Linear(N_HIDDEN, 1),

)



class LinReg(nn.Module):    

    def __init__(self, n_input, n_hidden, n_output):

        super(LinReg, self).__init__()    

        self.n_input=n_input

        self.n_hidden=n_hidden

        self.n_output= n_output 

                            

        linear1=torch.nn.Linear(n_input,n_hidden)

        torch.nn.init.xavier_uniform(linear1.weight)        

        

        linear2=torch.nn.Linear(n_hidden,1)

        torch.nn.init.xavier_uniform(linear2.weight)        

                

        self.classifier = torch.nn.Sequential(

#                                             linear1, nn.BatchNorm1d(n_hidden),dropout, relu,

                                            linear1,dropout, relu,

                                            linear2,              

                                  )                                                                 

    def forward(self, x):        

#         debug('x',x)           

        varSize=x.data.shape[0] # must be calculated here in forward() since its is a dynamic size                          

        x=x.contiguous() 

        x=self.classifier(x)                   

        return x

    

model=LinReg(N_FEATURES,N_HIDDEN,1)



print (model)

print(net_overfitting)

print(net_dropped)
criterion = torch.nn.MSELoss(size_average=True)

print (criterion)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print (optimizer)
LR = 0.005

BATCH_SIZE=32

EPOCH = 10 



import gc

df_train=None



gc.collect()





data_train = np.array(data_train, dtype=np.float32)



labels_train=labels_train.reshape((labels_train.shape[0],1)) # Must be reshaped for PyTorch!

labels_train = np.array(labels_train, dtype=np.float)



X_tensor = (torch.from_numpy(data_train)).type(torch.FloatTensor) # Note the conversion for pytorch

Y_tensor = (torch.from_numpy(labels_train)).type(torch.FloatTensor) # Note the conversion for pytorch    





import torch.utils.data as Data

dataset = Data.TensorDataset(data_tensor = X_tensor, target_tensor = Y_tensor)

loader = Data.DataLoader(dataset = dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0)



print (loader)


LR = 0.005

BATCH_SIZE=32

EPOCH = 10 



net_SGD = LinReg(N_FEATURES,N_HIDDEN,1)

net_Momentum = LinReg(N_FEATURES,N_HIDDEN,1)

net_RMSprop = LinReg(N_FEATURES,N_HIDDEN,1)

net_Adam = LinReg(N_FEATURES,N_HIDDEN,1)





opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr = LR)

opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr = LR, momentum = 0.9)

opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr = LR, alpha = 0.9)

opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr = LR, betas= (0.9, 0.99))



loss_func = torch.nn.MSELoss()



loss_SGD = []

loss_Momentum = []

loss_RMSprop =[]

loss_Adam = []



# losses = [loss_SGD, loss_Momentum, loss_RMSprop, loss_Adam]

losses = [loss_SGD, loss_Adam]

# nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

nets = [net_SGD, net_Adam]

# optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

optimizers = [opt_SGD, opt_Adam]



print (nets[0])



for epoch in range(0, EPOCH + 1):

    print('Training Epoch= {}/{} '.format(epoch,EPOCH))                    

    for step, (batch_x, batch_y) in enumerate(loader):

        var_x = Variable(batch_x)

        var_y = Variable(batch_y)

        for net, optimizer, loss_history in zip(nets, optimizers, losses): 

#             print ('Model:' + type(net).__name__) 

#             print ('Opt:' + type(optimizer).__name__)

            prediction = net(var_x)            

            loss = loss_func(prediction, var_y)            

            optimizer.zero_grad()            

            loss.backward()            

            optimizer.step()            

            loss_history.append(loss.data[0])

            

#     if epoch % 5  == 0:        

    loss_run = loss.data[0]                

    print(step, loss_run)               

    print('Training MSELoss=%.4f' % loss_run)                    
%matplotlib inline

import matplotlib.pyplot as plt



labels = ['SGD', 'Adam']

          

for i, loss_history in enumerate(losses):

    plt.plot(loss_history, label = labels[i])

          

plt.legend(loc = 'best')

plt.xlabel('Steps')

plt.ylabel('Loss')

plt.ylim((0, 0.2))

plt.show()
%matplotlib inline

import matplotlib.pyplot as plt

plt.plot(loss_history)

plt.show()



# # # ! pip install visdom



# # from visdom import Visdom

# # viz = Visdom()



# # num_epoch=int(epochs/div_factor)



# # x = np.reshape([i for i in range(num_epoch)],newshape=[num_epoch,1])

# # loss_data = np.reshape(loss_arr,newshape=[num_epoch,1])



# # win3=viz.line(

# #     X = x,

# #     Y = loss_data,

# #     opts=dict(

# #         xtickmin=0,

# #         xtickmax=num_epoch,

# #         xtickstep=1,

# #         ytickmin=0,

# #         ytickmax=20,

# #         ytickstep=1,

# #         markercolor=np.random.randint(0, 255, num_epoch),

# #     ),

# # )
# df_test.to_csv('test_clean.csv', header=False,  index = False)    

# df_test= pd.read_csv('test_clean.csv', header=None, dtype=np.float32)    

# feature_cols = list(df_train.columns[:-1])

# print (feature_cols)

# trainX = df_test[feature_cols]]


print('making predictions\n')

import pandas as pd

x=pd.read_csv ('../input/0609034-0608800-submission/0.609034_0.608800_submission.csv')

x.to_csv('sample_submission.csv', index=False)