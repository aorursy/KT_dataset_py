# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import warnings

warnings.filterwarnings('ignore')



#General Use

from sklearn import metrics

from sklearn.preprocessing import LabelEncoder

import matplotlib as mp

import matplotlib.pyplot as plt

import csv

import string

from datetime import datetime



# Regression

from sklearn.linear_model import LogisticRegression

from sklearn.grid_search import GridSearchCV



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
t_train = pd.read_csv('../input/train.csv')

t_test = pd.read_csv('../input/test.csv')

id_test = t_test['PassengerId']

labels = t_train['Survived'].values

t_train = t_train.drop(['Survived'], axis=1)

no_train = t_train.shape[0]



print (no_train)
# concatenate the train and test user files

tt_all = pd.concat((t_train, t_test), axis=0, ignore_index=True)



# drop columns unecessary for prediction

tt_all = tt_all.drop(['PassengerId', 'Name', 'Ticket','Fare','Cabin'], axis=1)



#set unknown gender values to NA

tt_all.Age = tt_all.Age.replace('',np.nan)



#fill NA values with -1

tt_all = tt_all.fillna(-1)



# encode categorical features with dummy values

categorical = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']

for f in categorical:

    tt_all_dummy = pd.get_dummies(tt_all[f], prefix=f)

    tt_all = tt_all.drop([f], axis=1)

    tt_all = pd.concat((tt_all, tt_all_dummy), axis=1)

    

print ("Done!")
# After cleansing, split the data back up between the train and test users

vals = tt_all.values

full_train = vals[:no_train]

full_test = vals[no_train:]



# Split training values between train & dev sets 



np.random.seed(0)

msk = np.random.rand(len(full_train)) < .75



split_train_data = full_train[msk]

split_train_labels = labels[msk]



split_dev_data = full_train[~msk]

split_dev_labels = labels[~msk]
def GenerateSubmission(preds):

    idx = np.empty(id_test.shape[0],)

    surv = np.empty(id_test.shape[0],)

    for i in range(id_test.shape[0]):

        idx[i] = id_test[i]

        surv[i] = preds[i]

    ## Output predictions to CSV

    subs = np.stack((idx, surv), axis = 1)

    print (subs.shape)

    sub = pd.DataFrame(data=subs, columns=['PassengerId', 'Survived'])

    print (sub.shape)

    sub.to_csv('submission.csv',index=False)
def LogReg():

    strengths = {'C': [0.0001,0.001,0.01,0.1,0.3,0.5,1.0]}



    # GridSearch for optimal regularization strength

    clf_lr = GridSearchCV(LogisticRegression(multi_class='ovr',penalty='l1'), strengths, scoring='f1_micro')



    clf_lr.fit(split_train_data, split_train_labels)



    # development predictions

    dev_preds = clf_lr.predict(split_dev_data)



    print ("Optimal Regularization Strength:", clf_lr.best_params_)

    print ("LogReg F1:", metrics.f1_score(split_dev_labels, dev_preds, average='micro'))

    

    t_preds = clf_lr.predict(full_test)

    print (t_preds.shape)

    return t_preds

    
t_preds = LogReg()

GenerateSubmission(t_preds)
