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
import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

# load datasets

train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')

trainlabels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
# examine features in each dataset

print("train :", train.keys())

print("test :", test.keys())

print("labels :", trainlabels.keys())

print("specs :", specs.keys())
# examine number of rows / observations in each dataset

print("train :", train.shape)

print("test :", test.shape)

print("labels :", trainlabels.shape)

print("specs :", specs.shape)
train.head()
trainlabels.head()
#count world

import collections

collections.Counter(train['world'])

collections.Counter(test['world'])
#plot success per assessment 

sns.barplot(x="accuracy_group", y="accuracy", data=trainlabels, hue="title")
collections.Counter(trainlabels['title'])
#plot time per world

sns.barplot(x="game_time", y="world", data=train, hue="title")
train_2 = pd.merge(train, trainlabels, on='game_session')

train_2.shape
#features in merged dataset 

train_2.keys()
#drop repeated columns in merged data (installation id and title features)

train_3 = train_2.drop(['installation_id_y', 'title_y'],1)



#rename

train_3 = train_2.rename(columns={"title_x": "title", "installation_id_x": "installation_id"})



#print keys

train_3.keys()
train_3.shape
#plot accuracy groups (0,1,2,3)

sns.countplot(x="accuracy_group", data=train_3)
sns.countplot(x="accuracy_group", data=train_3, hue="title")
#check unique values

train_2 = pd.get_dummies(train_3, columns=['event_code', 'world','title'], drop_first=True)

test_2 = pd.get_dummies(test, columns=['event_code', 'world','title'], drop_first=True)
print("train shape ", train_2.shape)

print("test shape ", test_2.shape)
#list of train features

train_list = train_2.keys()



#drop test feature if not in train list

test_3 = test_2.drop(columns=[col for col in test_2 if col not in train_list])



#print shapes

print("train shape ", train_2.shape) 

print("test shape ", test_3.shape)
from datetime import datetime

import time

#parse timestamp columns as timestamp data types

train_2['date'] = pd.to_datetime(train_2['timestamp']).astype('datetime64[ns]')

test_3['date'] = pd.to_datetime(test_3['timestamp']).astype('datetime64[ns]')
#hour and days column to codense



#create hour feature (0 - 24)

train_2['t_hour'] = (train_2['date']).dt.hour

test_3['t_hour'] = (test_3['date']).dt.hour



#create day feature (0-Sunday,..., 6-Saturday)

train_2['t_day'] = (train_2['date']).dt.week

test_3['t_day'] = (test_3['date']).dt.week 



#print shapes

print("train shape ", train_2.shape) 

print("test shape ", test_3.shape)
#delete useless features for model 



train_3 = train_2.drop(['date','event_id','game_session','installation_id','type','num_correct',

       'num_incorrect','accuracy','timestamp'], 1)

test_4 = test_3.drop(['date', 'event_id','game_session','installation_id','type','timestamp','event_data'], 1)



#shape

print("train shape ", train_3.shape) 

print("test shape ", test_4.shape)
#features for both train and test



#for train

train_X = train_3.drop(['accuracy_group','event_data','title_y','installation_id_y'], 1)

train_y = train_3['accuracy_group']



#for test

test_X = test_4
#simple decision tree 

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



dtree = DecisionTreeClassifier()

dtree_model = dtree.fit(train_X, train_y)





dtree_train_y = dtree_model.predict(train_X) 

dtree_val_y = dtree_model.predict(test_X) 

dtree_train_accuracy = accuracy_score(train_y, dtree_train_y)



result = confusion_matrix(dtree_train_y, train_y)

print("Confusion Matrix:")

print(result)

result1 = classification_report(dtree_train_y, train_y)

print("Classification Report:",)

print (result1)

result2 = accuracy_score(dtree_train_y, train_y)

print("Accuracy:",result2)
feature_names = ['event_count', 'game_time', 'event_code_2010', 'event_code_2020',

       'event_code_2025', 'event_code_2030', 'event_code_2035',

       'event_code_3010', 'event_code_3020', 'event_code_3021',

       'event_code_3110', 'event_code_3120', 'event_code_3121',

       'event_code_4020', 'event_code_4025', 'event_code_4030',

       'event_code_4035', 'event_code_4040', 'event_code_4070',

       'event_code_4080', 'event_code_4090', 'event_code_4100',

       'event_code_4110', 'world_MAGMAPEAK', 'world_TREETOPCITY',

       'title_Cart Balancer (Assessment)',

       'title_Cauldron Filler (Assessment)', 'title_Chest Sorter (Assessment)',

       'title_Mushroom Sorter (Assessment)', 't_hour', 't_day']
#Visualize Tree

#Too large

#estimator = dtree_model



#from sklearn.tree import export_graphviz

# Export as dot file

#export_graphviz(estimator, out_file='tree.dot', 

                #feature_names = feature_names,

                #class_names = feature_names,

                #rounded = True, proportion = False, 

                #precision = 2, filled = True)



# Convert to png

#from subprocess import call

#call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



# Display in python

#import matplotlib.pyplot as plt

#plt.figure(figsize = (14, 18))

#plt.imshow(plt.imread('tree.png'))

#plt.axis('off');

#plt.show();
test['accuracy_group'] = dtree_model.predict(test_X)



test.head()
import csv



done = test.loc[:,['installation_id','accuracy_group']]

submission = done.drop_duplicates(subset="installation_id", keep="last")
submission.to_csv('submission.csv', index=False)
submission.shape
#set up log-reg