import os
import json
import glob
import time
import copy
import torch
import random
import os.path
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.spatial.distance import pdist , cdist
from sklearn.metrics import f1_score
from sklearn import tree
from sklearn.model_selection import train_test_split
import torch.utils.data.dataset
from torch import optim
from sklearn.ensemble import RandomForestClassifier
test = pd.read_csv('../input/data-without-drift/test_clean.csv',header = 0)
train = pd.read_csv('../input/data-without-drift/train_clean.csv',header = 0)

#https://www.kaggle.com/cdeotte/one-feature-model-0-930
plt.figure(figsize=(20,5)); res = 1000
plt.plot(range(0,train.shape[0],res),train.signal[0::res])
for i in range(11): plt.plot([i*500000,i*500000],[-5,12.5],'r')
for j in range(10): plt.text(j*500000+200000,10,str(j+1),size=20)
plt.xlabel('Row',size=16); plt.ylabel('Signal',size=16); 
plt.title('Training Data Signal - 10 batches',size=20)
plt.show()
class FullDataset(torch.utils.data.Dataset):
    
    @property
    def data(self):
        return self.__data
    
    @property
    def label(self):
        return self.__label
    
    def __init__(self, data, mode):
        self.__mode=mode
        
        unique_label = list(data['open_channels'].unique())
        unique_label.sort()
        
        self.__weight = torch.zeros((11,), dtype=torch.float)
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()
        # split the data of each user between train_data and test_data
        for idx in unique_label:
            idx_data = data[data['open_channels']==idx]
            train, test = train_test_split(idx_data, test_size = 0.2, random_state = 10)
            self.__weight[int(idx)] = len(train)
            train_data = pd.concat([train_data, train])
            test_data = pd.concat([test_data, test])
        del idx_data, train, test, idx
          
        # reset the index of the dataframe
        train_data = train_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)
                
        X = train_data['signal']
        Y = train_data['open_channels']
        
        X_test = test_data['signal']
        Y_test = test_data['open_channels']

        if mode=='train':
            self.__data = X.values
            self.__label = Y.values
        elif mode=='test':
            self.__data = X_test.values
            self.__label = Y_test.values
        
    def __len__(self):
        return self.__data.shape[0]

    def __getitem__(self, index):                
        return torch.FloatTensor(np.array(self.__data[index])), int(self.__label[index])
train2 = train.copy()

batch = 1; a = 500000*(batch-1); b = 500000*batch
batch = 2; c = 500000*(batch-1); d = 500000*batch
X_train = np.concatenate([train2.signal.values[a:b],train2.signal.values[c:d]]).reshape((-1,1))
y_train = np.concatenate([train2.open_channels.values[a:b],train2.open_channels.values[c:d]]).reshape((-1,1))

dataset = pd.DataFrame(np.concatenate((X_train,y_train), axis=1), columns=['signal','open_channels'])

train_dataset = FullDataset(dataset, 'train')
test_dataset = FullDataset(dataset, 'test')
clf1=RandomForestClassifier(n_estimators=50)
# trainined with the 80% (train_dataset.data)
clf1.fit(np.expand_dims(train_dataset.data, axis=1), train_dataset.label)

# predict the label of the 20% (test_dataset.data)
y_pred=clf1.predict(np.expand_dims(test_dataset.data, axis=1))

print('has f1 validation score =',f1_score(test_dataset.label,y_pred,average='macro'))
batch = 3; a = 500000*(batch-1); b = 500000*batch
batch = 7; c = 500000*(batch-1); d = 500000*batch
X_train = np.concatenate([train2.signal.values[a:b],train2.signal.values[c:d]]).reshape((-1,1))
y_train = np.concatenate([train2.open_channels.values[a:b],train2.open_channels.values[c:d]]).reshape((-1,1))

dataset = pd.DataFrame(np.concatenate((X_train,y_train), axis=1), columns=['signal','open_channels'])
train_dataset = FullDataset(dataset, 'train')
test_dataset = FullDataset(dataset, 'test')
clf2=RandomForestClassifier(n_estimators=50)
clf2.fit(np.expand_dims(train_dataset.data, axis=1), train_dataset.label)
y_pred=clf2.predict(np.expand_dims(test_dataset.data, axis=1))
print('has f1 validation score =',f1_score(test_dataset.label,y_pred,average='macro'))
batch = 4; a = 500000*(batch-1); b = 500000*batch
batch = 8; c = 500000*(batch-1); d = 500000*batch
X_train = np.concatenate([train2.signal.values[a:b],train2.signal.values[c:d]]).reshape((-1,1))
y_train = np.concatenate([train2.open_channels.values[a:b],train2.open_channels.values[c:d]]).reshape((-1,1))


dataset = pd.DataFrame(np.concatenate((X_train,y_train), axis=1), columns=['signal','open_channels'])
train_dataset = FullDataset(dataset, 'train')
test_dataset = FullDataset(dataset, 'test')
clf3=RandomForestClassifier(n_estimators=50)

clf3.fit(np.expand_dims(train_dataset.data, axis=1), train_dataset.label)

y_pred=clf3.predict(np.expand_dims(test_dataset.data, axis=1))

print('has f1 validation score =',f1_score(test_dataset.label,y_pred,average='macro'))
batch = 6; a = 500000*(batch-1); b = 500000*batch
batch = 9; c = 500000*(batch-1); d = 500000*batch
X_train = np.concatenate([train2.signal.values[a:b],train2.signal.values[c:d]]).reshape((-1,1))
y_train = np.concatenate([train2.open_channels.values[a:b],train2.open_channels.values[c:d]]).reshape((-1,1))

dataset = pd.DataFrame(np.concatenate((X_train,y_train), axis=1), columns=['signal','open_channels'])
train_dataset = FullDataset(dataset, 'train')
test_dataset = FullDataset(dataset, 'test')
clf4=RandomForestClassifier(n_estimators=50)

clf4.fit(np.expand_dims(train_dataset.data, axis=1), train_dataset.label)

y_pred=clf4.predict(np.expand_dims(test_dataset.data, axis=1))

print('has f1 validation score =',f1_score(test_dataset.label,y_pred,average='macro'))
batch = 5; a = 500000*(batch-1); b = 500000*batch
batch = 10; c = 500000*(batch-1); d = 500000*batch
X_train = np.concatenate([train2.signal.values[a:b],train2.signal.values[c:d]]).reshape((-1,1))
y_train = np.concatenate([train2.open_channels.values[a:b],train2.open_channels.values[c:d]]).reshape((-1,1))

dataset = pd.DataFrame(np.concatenate((X_train,y_train), axis=1), columns=['signal','open_channels'])
train_dataset = FullDataset(dataset, 'train')
test_dataset = FullDataset(dataset, 'test')
clf5=RandomForestClassifier(n_estimators=50)

clf5.fit(np.expand_dims(train_dataset.data, axis=1), train_dataset.label)

y_pred=clf5.predict(np.expand_dims(test_dataset.data, axis=1))

print('has f1 validation score =',f1_score(test_dataset.label,y_pred,average='macro'))
test2 = test.copy()
let = ['A','B','C','D','E','F','G','H','I','J']
plt.figure(figsize=(20,5))
res = 1000
plt.plot(range(0,test2.shape[0],res),test2.signal[0::res])
for i in range(5): plt.plot([i*500000,i*500000],[-5,12.5],'r')
for i in range(21): plt.plot([i*100000,i*100000],[-5,12.5],'r:')
for k in range(4): plt.text(k*500000+250000,10,str(k+1),size=20)
for k in range(10): plt.text(k*100000+40000,7.5,let[k],size=16)
plt.title('Test Signal without Drift',size=16)
plt.show()
sub = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv', dtype={'time': 'str'})


a = 0 # SUBSAMPLE A, Model 1
sub.iloc[100000*a:100000*(a+1),1] = clf1.predict(test2.signal.values[100000*a:100000*(a+1)].reshape((-1,1)))

a = 1 # SUBSAMPLE B, Model 3
sub.iloc[100000*a:100000*(a+1),1] = clf3.predict(test2.signal.values[100000*a:100000*(a+1)].reshape((-1,1)))

a = 2 # SUBSAMPLE C, Model 5
sub.iloc[100000*a:100000*(a+1),1] = clf4.predict(test2.signal.values[100000*a:100000*(a+1)].reshape((-1,1)))

a = 3 # SUBSAMPLE D, Model 1
sub.iloc[100000*a:100000*(a+1),1] = clf1.predict(test2.signal.values[100000*a:100000*(a+1)].reshape((-1,1)))

a = 4 # SUBSAMPLE E, Model 2
sub.iloc[100000*a:100000*(a+1),1] = clf2.predict(test2.signal.values[100000*a:100000*(a+1)].reshape((-1,1)))

a = 5 # SUBSAMPLE F, Model 5
sub.iloc[100000*a:100000*(a+1),1] = clf5.predict(test2.signal.values[100000*a:100000*(a+1)].reshape((-1,1)))

a = 6 # SUBSAMPLE G, Model 4
sub.iloc[100000*a:100000*(a+1),1] = clf4.predict(test2.signal.values[100000*a:100000*(a+1)].reshape((-1,1)))

a = 7 # SUBSAMPLE H, Model 5
sub.iloc[100000*a:100000*(a+1),1] = clf5.predict(test2.signal.values[100000*a:100000*(a+1)].reshape((-1,1)))

a = 8 # SUBSAMPLE I, Model 1
sub.iloc[100000*a:100000*(a+1),1] = clf1.predict(test2.signal.values[100000*a:100000*(a+1)].reshape((-1,1)))

a = 9 # SUBSAMPLE J, Model 3
sub.iloc[100000*a:100000*(a+1),1] = clf3.predict(test2.signal.values[100000*a:100000*(a+1)].reshape((-1,1)))

 # BATCHES 3 AND 4, Model 1
sub.iloc[1000000:2000000,1] = clf1.predict(test2.signal.values[1000000:2000000].reshape((-1,1)))
plt.figure(figsize=(20,5))
res = 1000
plt.plot(range(0,test.shape[0],res),sub.open_channels[0::res])
for i in range(5): plt.plot([i*500000,i*500000],[-5,12.5],'r')
for i in range(21): plt.plot([i*100000,i*100000],[-5,12.5],'r:')
for k in range(4): plt.text(k*500000+250000,10,str(k+1),size=20)
for k in range(10): plt.text(k*100000+40000,7.5,let[k],size=16)
plt.title('Test Data Predictions',size=16)
plt.show()
sub['open_channels'] = sub['open_channels'].astype(int)
sub.to_csv('submission.csv',index=False,float_format='%.4f')
#rm -r /kaggle/working/my_submission.csv
