import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

import time

import datetime

import gc

import random

import re

import operator

import pickle

from tqdm import tqdm



from sklearn.model_selection import StratifiedKFold,KFold

from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,log_loss



import torch

import torch.nn as nn

from torch.autograd import Variable

from torch.utils.data import DataLoader,TensorDataset,Dataset

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.optim.optimizer import Optimizer



from keras.preprocessing.text import Tokenizer,text_to_word_sequence

from keras.preprocessing.sequence import pad_sequences



def seed_everything(SEED=42):

    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)

    torch.cuda.manual_seed(SEED)

#     torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True

    os.environ['PYTHONHASHSEED']=str(SEED)

    # torch.backends.cudnn.benchmark = False



def init_func(worker_id):

    np.random.seed(SEED+worker_id)



tqdm.pandas()

SEED=42

seed_everything(SEED=SEED)



# noting down the run time of the kernel

t1=datetime.datetime.now()
X_train = np.hstack((np.arange(120).reshape((-1,1)),np.random.normal(0,1,size=(120,30))))

y_train = np.hstack((np.arange(120).reshape((-1,1)),np.random.binomial(1,0.4,size=((120,1)))))

weights = np.ones(120,)



# validation id starts from 100

X_val = np.hstack((np.arange(20).reshape((-1,1)) + 100,np.random.normal(0,1,size=(20,30))))

y_val = np.hstack((np.arange(20).reshape((-1,1)) + 100,np.random.binomial(1,0.4,size=(20,1))))



# test id start from 120

X_test = np.hstack((np.arange(40).reshape((-1,1)) + 120,np.random.normal(0,1,size=(40,30))))



print(X_train.shape,X_val.shape,X_test.shape)

print(y_train.shape,y_val.shape)
def make_train_dataset(X_train,y_train,batch_size=32,weights=None):

    """

        weights : weights of training examples. length of m_train

    """

    if weights is None:

        weights=np.ones((y_train.shape[0],))

    X_train,y_train,weights=torch.Tensor(X_train),torch.Tensor(y_train),torch.Tensor(weights)

    

    # train dataset 

    train_dataset=TensorDataset(X_train,y_train,weights)

    

    # Passing this to data loader

    # shuffle set to true imples for every epoch data is shuffled

    train_iterator=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)

    

    return train_iterator





def make_val_dataset(X_val,y_val,batch_size=32):

    X_val,y_val=torch.Tensor(X_val),torch.Tensor(y_val)

    

    # val dataset

    val_dataset=TensorDataset(X_val,y_val)

    

    # Passing this to data loader

    val_iterator=DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=0)

    

    return val_iterator



def make_test_dataset(X_test,batch_size=32):

    # making test data and test iterator

    if len(y_train.shape) > 1 :

        y_test = np.zeros((X_test.shape[0],y_train.shape[1]))

    else:

        y_test = np.zeros((X_test.shape[0],))

    

    X_test,y_test = torch.Tensor(X_test),torch.Tensor(y_test)

    

    # test dataset

    test_dataset = TensorDataset(X_test,y_test)

    

    # passing this to data loader

    test_iterator = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=0)

    



    return test_iterator
for fold,(train_index,val_index) in  enumerate(StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED).\

                                                    split(X_train,y_train[:,1])):

    

    

    train_iterator = make_train_dataset(X_train[train_index],y_train[train_index],weights=weights[train_index])

    val_iterator = make_val_dataset(X_train[val_index],y_train[val_index])

    test_iterator = make_test_dataset(X_test)



    prev_rng_state = torch.get_rng_state()  # get previous rng state

    print("============================== FOLD = ",fold+1,"==========================")

    for ep_num in range(3):

        print("================================== EPOCH = ",ep_num+1,"========================")



        torch.set_rng_state(prev_rng_state) # set rng state

        for batch,(a,b,c) in enumerate(train_iterator):

            if batch==0:

                print("15 examples of train")

                print(a[0:15, 0])



        for batch,(a,b) in enumerate(val_iterator):

            if batch==0:

                print("15 examples of validation")

                print(a[0:15,0])



        prev_rng_state = torch.get_rng_state() # save rng state



#         for batch,(a,b) in enumerate(test_iterator):

#             if batch==0:

#                 print("15 examples of test")

#                 print(a[0:15,0])

                

        