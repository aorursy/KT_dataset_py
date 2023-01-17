import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import torch

import torch.nn as nn

import pandas as pd

import numpy as np

import xgboost as xgb

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.model_selection import TimeSeriesSplit,KFold, StratifiedKFold

from time import time

import pickle

import datetime as dt

from imblearn.over_sampling import RandomOverSampler

from scipy.sparse import csc_matrix
import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
# read data

def init_read_big(train_path, test_path, train_big_path):

    train = pd.read_csv(train_path)

    test = pd.read_csv(test_path)

    train_big = pd.read_csv(train_big_path)

    return train, test, train_big



def init_read_small(train_path, test_path):

    train = pd.read_csv(train_path)

    test = pd.read_csv(test_path)

    return train, test



def remove_column_big(train, train_big, column_title_big):

    

    #train = train.drop(['Quote_ID'], axis=1)

    train_small = train.copy()

    column_title_small = [A for A in train.columns]

    train_big = train_big[column_title_big]

    train = train_big.copy()

    train.columns = column_title_small

    

    return train

    

    



def data_preprocessing(train, test):

    train = train.drop(['Quote_ID'], axis=1)

    

    y = train.QuoteConversion_Flag.values

    

    train = train.drop(['QuoteConversion_Flag'], axis=1)

    test = test.drop('Quote_ID', axis=1)

    

    

    train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))

    train = train.drop('Original_Quote_Date', axis=1)

    

    test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))

    test = test.drop('Original_Quote_Date', axis=1)

    

    

    #train['Date_value'] = train['Date'].apply(lambda x: float((x-dt.datetime(2009, 12, 30)).days)+(float((x-dt.datetime(2009, 12, 30)).seconds)/86400))

    #test['Date_value'] = test['Date'].apply(lambda x: float((x-dt.datetime(2009, 12, 30)).days)+(float((x-dt.datetime(2009, 12, 30)).seconds)/86400))

    

    

    train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))

    train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))

    train['weekday'] = train['Date'].dt.dayofweek



    test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))

    test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))

    test['weekday'] = test['Date'].dt.dayofweek

    

    # count the number of missing values in each row

    #train['miss_count'] = train.apply(lambda x: len(train.columns)- x.count(), axis=1)

    #test['miss_count'] = test.apply(lambda x: len(test.columns)- x.count(), axis=1)

    

    # drop redundant attributes

    #train = train.drop(['Date','Property_info2','Field_info1'], axis=1)

    #test = test.drop(['Date','Property_info2','Field_info1'], axis=1)

    

    train = train.drop(['Date'], axis=1)

    test = test.drop(['Date'], axis=1)

    

    

    '''

    # use one-hot encoding for categorical attribute

    

    onehot_columns =  [

                'Field_info1',

                'Field_info3',                

                'Coverage_info3',

                'Sales_info4',

                'Geographic_info5']

    

    labelencoder_columns = ['Field_info4','Personal_info1','Property_info1',

                            'Geographic_info4','Personal_info3','Property_info3']

    

    

    for f in labelencoder_columns:

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train[f].values) + list(test[f].values))

        train[f] = lbl.transform(list(train[f].values))

        test[f] = lbl.transform(list(test[f].values))

        

    train = pd.get_dummies(train, prefix_sep="__",columns=onehot_columns)

    test = pd.get_dummies(test, prefix_sep="__",columns=onehot_columns)

   

    '''

    

    # simplying convert all categorical attributes to number

    for A in train.columns:

        if train[A].dtype=='object':

            lbl = preprocessing.LabelEncoder()

            lbl.fit(list(train[A].values) + list(test[A].values))

            train[A] = lbl.transform(list(train[A].values))

            test[A] = lbl.transform(list(test[A].values))

    

    

    # fill missing values for Personal_info5

    #train["Personal_info5"] = train["Personal_info5"].fillna(0)

    #test["Personal_info5"] = test["Personal_info5"].fillna(0)

    

    

    train = train.fillna(-999)

    test = test.fillna(-999)

    

    

    

    print("\nPre-processing_big complete!!")

    

    return train, test, y

train_path = '/kaggle/input/2019s-uts-data-analytics-assignment-3/Assignment3_TrainingSet.csv'

test_path = '/kaggle/input/2019s-uts-data-analytics-assignment-3/Assignment3_TestSet.csv'

train, test = init_read_small(train_path, test_path)

train, test, y = data_preprocessing(train, test)
'''

from imblearn.over_sampling import RandomOverSampler



ros = RandomOverSampler()

train, y = ros.fit_sample(train, y)

'''
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

train = sc.fit_transform(train)

test = sc.fit_transform(test)
'''

X_train, X_valid, y_train, y_valid = train_test_split(train, y, 

                                    test_size = 0.2,random_state = 999)



X_train_torch = torch.tensor(X_train)

X_valid_torch = torch.tensor(X_valid)

y_train_torch = torch.tensor(y_train)

y_valid_torch = torch.tensor(y_valid)

test_torch = torch.tensor(test)



# print out the size info

print(X_train_torch.size())

print(X_valid_torch.size())

print(y_train_torch.size())

print(y_valid_torch.size())



'''

class Net(nn.Module):



    def __init__(self, D_in, H=15, D_out=1):

        super().__init__()

        self.fc1 = nn.Linear(D_in, H)

        self.fc2 = nn.Linear(H, D_out)

        self.relu = nn.ReLU()



    def forward(self, x):

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)



        return x.squeeze()
class EarlyStopping(object):

    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):

        self.mode = mode

        self.min_delta = min_delta

        self.patience = patience

        self.best = None

        self.num_bad_epochs = 0

        self.is_better = None

        self._init_is_better(mode, min_delta, percentage)



        if patience == 0:

            self.is_better = lambda a, b: True

            self.step = lambda a: False



    def step(self, metrics):

        if self.best is None:

            self.best = metrics

            return False



        if np.isnan(metrics):

            return True



        if self.is_better(metrics, self.best):

            self.num_bad_epochs = 0

            self.best = metrics

        else:

            self.num_bad_epochs += 1



        if self.num_bad_epochs >= self.patience:

            return True



        return False



    def _init_is_better(self, mode, min_delta, percentage):

        if mode not in {'min', 'max'}:

            raise ValueError('mode ' + mode + ' is unknown!')

        if not percentage:

            if mode == 'min':

                self.is_better = lambda a, best: a < best - min_delta

            if mode == 'max':

                self.is_better = lambda a, best: a > best + min_delta

        else:

            if mode == 'min':

                self.is_better = lambda a, best: a < best - (

                            best * min_delta / 100)

            if mode == 'max':

                self.is_better = lambda a, best: a > best + (

                            best * min_delta / 100)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_fold = 5

random_state = 999

D_in, H = 30, 15



models = []

train_no = 1

training_cycle = 20000

repetition = 1

training_sequence = 1





for A in range(repetition):

    

    kf =  StratifiedKFold(n_splits = n_fold , shuffle = True, random_state = random_state+A)

    

    for train_index, val_index in kf.split(train, y):

        train_X = train[train_index]

        val_X = train[val_index]

        train_y = y[train_index]

        val_y = y[val_index]

        

        train_X_torch = torch.tensor(train_X)

        val_X_torch = torch.tensor(val_X )

        train_y_torch = torch.tensor(train_y)

        val_y_torch = torch.tensor(val_y)



        print(f'\n\nurrently Training sequence no {training_sequence}')

        

    

        net = Net(D_in, H).to(device)

        criteria = nn.MSELoss()

        optimizer = optim.Adam(net.parameters(), weight_decay=0.0001)

        

        es = EarlyStopping(patience=20)

       

        for epoch in range(training_cycle):



            print(f'Training Epoch No {epoch}')



            inputs = train_X_torch.to(device)

            labels = train_y_torch.to(device)



            inputs_valid = val_X_torch.to(device)

            labels_valid = val_y_torch.to(device)





            # Zero the parameter gradients

            optimizer.zero_grad()



            # Forward + backward + optimize

            outputs = net(inputs.float())

            outputs_valid = net(inputs_valid.float())

            loss = criteria(outputs, labels.float())

            loss_valid = criteria(outputs_valid, labels_valid.float())

            metric_valid = loss_valid.detach().numpy()

            loss.backward()

            optimizer.step()



            print(f'Current Loss is {metric_valid}')



            if es.step(metric_valid):

                break



        models.append(net)

        training_sequence += 1

        

        del net, es, metric_valid

  

'''

# Split into training and test

train_size = int(0.8 * len(train_torch))

test_size = len(train_torch) - train_size

trainset, testset = random_split(train_torch, [train_size, test_size])

train_y, test_y = random_split(y_torch, [train_size, test_size])

'''
'''

# Dataloaders

trainloader = DataLoader(trainset, batch_size=200, shuffle=True)

testloader = DataLoader(testset, batch_size=200, shuffle=False)

'''
'''

D_in, H = 30, 15

net = Net(D_in, H).to(device)



# Loss function

criteria = nn.MSELoss()

#criterion = nn.CrossEntropyLoss()



# Optimizer

optimizer = optim.Adam(net.parameters(), weight_decay=0.0001)





n_epochs = 20000

es = EarlyStopping(patience=20)

for epoch in range(n_epochs):

    

    print(f'Training Epoch No {epoch}')

    



    inputs = X_train_torch.to(device)

    labels = y_train_torch.to(device)

    

    inputs_valid = X_valid_torch.to(device)

    labels_valid = y_valid_torch.to(device)





    # Zero the parameter gradients

    optimizer.zero_grad()



    # Forward + backward + optimize

    outputs = net(inputs.float())

    outputs_valid = net(inputs_valid.float())

    loss = criteria(outputs, labels.float())

    loss_valid = criteria(outputs_valid, labels_valid.float())

    metric_valid = loss_valid.detach().numpy()

    loss.backward()

    optimizer.step()

    

    print(f'Current Loss is {metric_valid}')

    

    if es.step(metric_valid):

        break

        

        

'''

test_torch = torch.tensor(test)

pred_df = sum([net(test_torch.float()) for net in models])/(n_fold*repetition)
test_predict_numpy = pred_df.detach().numpy()
for A in range(len(test_predict_numpy)):

    if test_predict_numpy[A] > np.mean(test_predict_numpy):

        test_predict_numpy[A] = 1

    else:

        test_predict_numpy[A] = 0
sum(test_predict_numpy)
sample = pd.read_csv('/kaggle/input/2019s-uts-data-analytics-assignment-3/Assignment3_Random_Submission-Kaggle.csv')

sample.QuoteConversion_Flag = test_predict_numpy

sample.to_csv('pytorch_model.csv', index=False)

print("#########SUBMISSION FILE UPLOADED!!! WELL DONE!!!#########")
sample