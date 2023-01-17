# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import additional modules for visualization (exploratory data analysis)



import matplotlib.pyplot as plt

import seaborn as sns
# load all datasets

dsb_train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

dsb_test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')

dsb_trainlabels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

dsb_specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')



# i've had to make this edit because of cpu usage on kaggle

dsb_train = dsb_train.drop(['event_data'], 1)

dsb_test = dsb_test.drop(['event_data'], 1)
# examine features in each dataset

print("train :", dsb_train.keys())

print("test :", dsb_test.keys())

print("labels :", dsb_trainlabels.keys())

print("specs :", dsb_specs.keys())
# examine number of rows / observations in each dataset

print("train :", dsb_train.shape)

print("test :", dsb_test.shape)

print("labels :", dsb_trainlabels.shape)

print("specs :", dsb_specs.shape)
dsb_train.head()
dsb_trainlabels.head()
# import module for counting

import collections

collections.Counter(dsb_train['game_session'])
collections.Counter(dsb_trainlabels['game_session'])
new_train = pd.merge(dsb_train, dsb_trainlabels, on='game_session')
new_train.shape
# list features in merged dataset 

new_train.keys()
# drop repeated columns in merged data (installation id and title features)

new_train = new_train.drop(['installation_id_y', 'title_y'],1)



# rename installation_id_x and title_x

new_train = new_train.rename(columns={"title_x": "title", "installation_id_x": "installation_id"})



# check to see if renaming worked

new_train.keys()
# create pairplot to identify interesting trends

# sns.pairplot(new_train, hue="title") had to comment this out because of low computing power
# distribution of accuracy groups (0,1,2,3)

sns.countplot(x="accuracy_group", data=new_train)
sns.countplot(x="accuracy_group", data=new_train, hue="title")
# check unique values

#print(dsb_test['world'].unique())

#print(dsb_train['world'].unique())

#print(dsb_test['event_code'].unique())

#print(new_train['event_code'].unique())

#print(dsb_trainlabels['event_code'].unique()) etc
train2 = pd.get_dummies(new_train, columns=['event_code', 'world','title'], drop_first=True)

test2 = pd.get_dummies(dsb_test, columns=['event_code', 'world','title'], drop_first=True)



# quick note: drop_first is True if we want to drop the original column we are converting.
print("train shape ", train2.shape) # you can also use .keys() depending on what you are looking out for

print("test shape ", test2.shape)
# list of train features

train_list = train2.keys()



# next, drop test feature if not in train list

test3 = test2.drop(columns=[col for col in test2 if col not in train_list])



# print shapes to check if dropping worked

print("train shape ", train2.shape) 

print("test shape ", test3.shape)
# import date and time models

from datetime import datetime

import time
# parse timestamp columns as timestamp dtypes

train2['date'] = pd.to_datetime(train2['timestamp']).astype('datetime64[ns]')

test3['date'] = pd.to_datetime(test3['timestamp']).astype('datetime64[ns]')
# create new columns: hour and days



# 1. create hour feature (0 - 24)

train2['t_hour'] = (train2['date']).dt.hour

test3['t_hour'] = (test3['date']).dt.hour



# 2. create day feature (0-Sunday,..., 6-Saturday)

train2['t_day'] = (train2['date']).dt.weekday

test3['t_day'] = (test3['date']).dt.weekday 



# print shapes to check if we are on track

print("train shape ", train2.shape) 

print("test shape ", test3.shape)
# These are the features I don't believe are useful for our simple analysis



train3 = train2.drop(['date','event_id','game_session','installation_id','type','num_correct',

       'num_incorrect','accuracy','timestamp'], 1)

test4 = test3.drop(['date', 'event_id','game_session','installation_id','type','timestamp'], 1)



# print shapes to check if we are on track

print("train shape ", train3.shape) 

print("test shape ", test4.shape)
sns.set(font_scale=1.5) #increast size of plot font

plt.figure(figsize=(20, 10)) #increase size of plot

sns.heatmap(train3[['accuracy_group','title_Cart Balancer (Assessment)',

       'title_Cauldron Filler (Assessment)', 'title_Chest Sorter (Assessment)',

       'title_Mushroom Sorter (Assessment)', 't_hour', 't_day']].corr(), annot = True)
# Select X and y features for both train and test



# for train

train_X = train3.drop(['accuracy_group'], 1)

train_y = train3['accuracy_group']



# for test

test_X = test4
# DECISION TREE MODEL

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



dtree = DecisionTreeClassifier()

dtree_model = dtree.fit(train_X, train_y)



dtree_train_y = dtree_model.predict(train_X) #The Decision tree prediction for the train_X data.

dtree_val_y = dtree_model.predict(test_X) #The Decision tree prediction for the val_X data.

dtree_train_accuracy = accuracy_score(train_y, dtree_train_y) #The accuracy for the dtree_train_y prediction.



# Print Accuracies for Decision Tree

print("Decision Tree Training Accuracy: ", dtree_train_accuracy)
# i've had to comment this out as well due to cpu usage

"""

# K NEAREST NEIGHBOR MODEL

from sklearn.neighbors import KNeighborsClassifier



knn_model = KNeighborsClassifier() # you can opt to qualify number of neighbors here (eg. n_neighbors = 5)

knn_model.fit(train_X, train_y)



#This creates the prediction. 

knn_train_y = knn_model.predict(train_X) #The KNN prediction for the train_X data.

knn_val_y = knn_model.predict(test_X) #The KNN prediction for the val_X data.

knn_train_accuracy = accuracy_score(train_y, knn_train_y) #The accuracy for the knn0_train_y prediction.



# Print Accuracies for Decision Tree

print("KNN Training Accuracy: ", knn_train_accuracy)

"""
dsb_test['accuracy_group'] = dtree_model.predict(test_X)



# check test dataframe to see if appending worked

dsb_test.head()
import csv



# create dataframe

sub = dsb_test.loc[:,['installation_id','accuracy_group']]

submission = sub.drop_duplicates(subset="installation_id", keep="last") # dropping duplicates in test data
# create csv file from submission dataframe

submission.to_csv('submission.csv', index=False)
submission.shape