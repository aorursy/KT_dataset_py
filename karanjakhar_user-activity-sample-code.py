import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

import warnings 

from sklearn.utils import shuffle

from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

os.listdir('../input/testtraindata/')
base_train_dir = '../input/testtraindata/Train_Set/'

base_test_dir = '../input/testtraindata/Test_Set/'
test_data = pd.DataFrame(columns = ['activity','ax','ay','az','gx','gy','gz'])

files = os.listdir(base_test_dir)

for f in files:

    df = pd.read_csv(base_test_dir+f)

    df['activity'] = f.split('.')[0].split('_')[-1]

    test_data = pd.concat([test_data,df],axis = 0)
test_data = shuffle(test_data)

test_data.reset_index(drop = True,inplace = True)

test_data.head()
train_data = pd.DataFrame(columns = ['activity','ax','ay','az','gx','gy','gz'])

train_folders = os.listdir(base_train_dir)



for tf in train_folders:

    files = os.listdir(base_train_dir+tf)

    for f in files:

        df = pd.read_csv(base_train_dir+tf+'/'+f)

        train_data = pd.concat([train_data,df],axis = 0)

    
train_data = shuffle(train_data)

train_data.reset_index(drop = True,inplace = True)

train_data.head()
train_data['activity'] = train_data['activity'].str.strip()

train_dict = {'standing':0,'sitting':1,'walking':2,'standing':3,'lying':4}

train_data['activity'] = train_data['activity'].replace(train_dict)

test_data['activity'] = test_data['activity'].replace(train_dict)
test_data.head()
train_data.head()
x_train = train_data.drop('activity',axis = 1)

y_train = train_data['activity']

x_test = test_data.drop('activity',axis = 1)

y_test = test_data['activity']
clf = DecisionTreeClassifier()

clf.fit(x_train,y_train)

print('Accuracy :',clf.score(x_test,y_test))