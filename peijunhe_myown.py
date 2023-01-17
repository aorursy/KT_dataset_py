# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as skl
from sklearn.decomposition import PCA
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegressionCV

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
path = "../input/"
files = os.listdir("../input")

# Any results you write to the current directory are saved as output.
original_dataset = pd.read_csv(path+files[0])
original_features = list(original_dataset.columns)
original_dataset.dropna(inplace = True)   
original_dataset = original_dataset.sample(frac = 1).reset_index(drop = True)

length = original_dataset.shape[0]
trainlength = round(length * 0.7)
train_set = pd.DataFrame(original_dataset.iloc[0:trainlength], columns = original_features).reset_index(drop = True)
val_set = pd.DataFrame(original_dataset.iloc[trainlength:], columns = original_features).reset_index(drop = True)
test_set = pd.read_csv(path + files[1])

majors = ['squad','duo','solo','squad-fpp','duo-fpp','solo-fpp']
index1 = []
index2 = []
index3 = []

train_set.drop("winPlacePerc", axis = 1, inplace = True)
for i in range(train_set.shape[0]):
    if train_set['matchType'][i] not in majors:
        index1.append(i)

val_set.drop("winPlacePerc", axis = 1, inplace = True)
for j in range(val_set.shape[0]):
    if val_set['matchType'][j] not in majors:
        index2.append(j)        

for k in range(test_set.shape[0]):
    if test_set['matchType'][k] not in majors:
        index3.append(k)

train_set.drop(index1,inplace=True)
train_matchType = train_set['matchType']
for fea in list(train_set.columns):
    if type(train_set[fea].iloc[0]) == str:
        train_set.drop(fea, axis = 1, inplace = True)

val_set.drop(index2,inplace=True)
val_matchType = val_set['matchType']
for fea in list(val_set.columns):
    if type(val_set[fea].iloc[0]) == str:
        val_set.drop(fea, axis = 1, inplace = True)
        
test_set.drop(index3,inplace=True)
test_matchType = test_set['matchType']
for fea in list(test_set.columns):
    if type(test_set[fea].iloc[0]) == str:
        test_set.drop(fea, axis = 1, inplace = True)
mean = {}
std = {}
for col in train_set.columns:
    mean[col] = np.mean(train_set[col])
    std[col] = np.std(train_set[col])
    train_set[col] = (train_set[col] - mean[col]) / std[col]
    val_set[col] = (val_set[col] - mean[col]) / std[col]
    test_set[col] = (test_set[col] - mean[col]) / std[col]
d = {'squad-fpp': 20000, 'duo-fpp': 20000, 'squad': 20000, 'solo-fpp': 20000, 'solo': 20000, 'duo': 20000}
rus = RandomUnderSampler(sampling_strategy=d)
new_set, new_matchType = rus.fit_resample(train_set, train_matchType)
LRC = LogisticRegressionCV(penalty='l1',cv=5,solver='saga',multi_class='multinomial',tol=0.01,n_jobs=-1,max_iter=150).fit(train_set, train_matchType)
error1 = 1 - LRC.score(val_set, val_matchType)
print('Validation error for Logistic Regression:', error1)
DTC = DecisionTreeClassifier()
DTC.fit(train_set, train_matchType)
error2 = 1 - DTC.score(val_set, val_matchType)
print('Validation error for Decision Tree:', error2)
for i in range(1,11):
    n = i*5
    RFC = RandomForestClassifier(n_jobs=-1,n_estimators=n)
    RFC.fit(train_set,train_matchType)
    error3 = 1 - RFC.score(val_set, val_matchType)
    print('When the number of trees is', n)
    print('Validation error for Random Forest:', error3)
test_error = 1 - DTC.score(test_set, test_matchType)
print('Test error using Decision tree:', test_error)
RFC = RandomForestClassifier(n_jobs=-1,n_estimators=30)
RFC.fit(train_set,train_matchType)
test_error = 1 - RFC.score(test_set, test_matchType)
print('Test error using Random forest:', test_error)
