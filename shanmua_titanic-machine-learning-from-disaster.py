# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_data.head()
pd.DataFrame(train_data).info()
test_data.head()
pd.DataFrame(test_data).info()
train_Y = train_data['Survived']
train_X = train_data.drop(labels = ["Survived","PassengerId","Name"],axis = 1).values
pd.DataFrame(train_X).head()
test_X =test_data.drop(labels = ['PassengerId','Name'],axis = 1).values
pd.DataFrame(test_X).head()
#male = 0,female = 1
for i in range(0,891):
    if train_X[i,1] == 'male':
        train_X[i,1] = 0
    else:
        train_X[i,1] = 1
train_X[::,1].sum()
#male = 0,female = 1
for i in range(0,418):
    if test_X[i,1] == 'male':
        test_X[i,1] = 0
    else:
        test_X[i,1] = 1
test_X[::,1].sum()
import seaborn as sns
plt.figure(figsize=(12,6))
sns.countplot(x = "Sex",data = train_data,hue = 'Survived')
plt.figure(figsize = (12,6))
sns.countplot(x = "Embarked",data = train_data,hue = 'Survived')
for i in range(0,891):
    if train_X[i,8] == 'Q':
        train_X[i,8] = 0
    elif train_X[i,8] == 'C':
        train_X[i,8] = 1
    else:
        train_X[i,8] = 2
pd.DataFrame(train_X[::,8].reshape(1,-1)).head()
for i in range(0,418):
    if test_X[i,8] == 'Q':
        test_X[i,8] = 0
    elif train_X[i,8] == 'C':
        test_X[i,8] = 1
    else:
        test_X[i,8] = 2
pd.DataFrame(test_X[::,8].reshape(1,-1)).head()
#use re model
import re
pattern = '.*[A-Z|a-z].*'
for i in range(0,891):
    if re.search(pattern,train_X[i,5]):
        train_X[i,5] = 0
    else:
        train_X[i,5] = 1
pd.DataFrame(train_X[::,5].reshape(1,-1)).head()
plt.figure(figsize = (12,6))
sns.countplot(x=train_X[0,5],data = pd.DataFrame(train_X[::,5]),hue = train_data['Survived'])
pattern = '.*[A-Z|a-z].*'
for i in range(0,418):
    if re.search(pattern,test_X[i,5]):
        test_X[i,5] = 0
    else:
        test_X[i,5] = 1
pd.DataFrame(test_X[::,5].reshape(1,-1)).head()
#age
plt.figure(figsize = (40,6))
sns.countplot(x = 'Age', data = train_data, hue = 'Survived')
pd.DataFrame(train_X[::,2]).sum()
from scipy import stats
mean = pd.DataFrame(train_X[::,2]).sum() // (891 - pd.DataFrame(train_X[::,2]).isnull().sum())
median = np.median(train_X[::,2])
argmax = stats.mode(train_X[::,2])[0][0]
print(float(mean),median,argmax)
import math
flag = 0
for i in range(0,891):
    if math.isnan(train_X[i,2]):
        if flag == 0:
            train_X[i,2] = mean
            flag += 1
        elif flag == 1:
            train_X[i,2] = median
            flag += 1
        else:
            train_X[i,2] = argmax
            flag = 0
pd.DataFrame(train_X[::,2]).isnull().sum()
#test_X age
t_mean = pd.DataFrame(test_X[::,2]).sum() // (891 - pd.DataFrame(test_X[::,2]).isnull().sum())
t_median = np.median(test_X[::,2])
t_argmax = stats.mode(test_X[::,2])[0][0]
print(float(t_mean),t_median,t_argmax)
t_flag = 0
for i in range(0,418):
    if math.isnan(test_X[i,2]):
        if flag == 0:
            test_X[i,2] = t_mean
            flag += 1
        elif flag == 1:
            test_X[i,2] = t_median
            flag += 1
        else:
            test_X[i,2] = t_argmax
            flag = 0
pd.DataFrame(test_X[::,2]).isnull().sum()
pd.DataFrame(train_X[::,7]).isnull().sum()
#Fare
Fare_flag = pd.DataFrame(train_X[::,7].reshape(1,-1)).isnull()
Fare_flag.head()
for i in range(0,891):
    if Fare_flag[i].bool():
        train_X[i,7] = 0
    else:
        train_X[i,7] = 1
pd.DataFrame(train_X[::,7].reshape(1,-1)).head()
pd.DataFrame(test_X[::,7]).isnull().sum()
t_Fare_flag = pd.DataFrame(test_X[::,7].reshape(1,-1)).isnull()
t_Fare_flag.head()
for i in range(0,418):
    if t_Fare_flag[i].bool():
        test_X[i,7] = 0
    else:
        test_X[i,7] = 1
pd.DataFrame(test_X[::,7].reshape(1,-1)).head()
pd.DataFrame(train_X).head()
pd.DataFrame(test_X).head()
pd.DataFrame(train_X).info()
pd.DataFrame(test_X).info()
for i in range(0,418):
    if math.isnan(test_X[i,6]):
        test_X[i,6] = test_X[i - 1,6]
pd.DataFrame(test_X).info()