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
%matplotlib inline

#Use matplotlib and seaborn packages to create beautiful visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support
train_set = pd.read_csv('../input/train.csv')
train_set.shape
train_set.columns
# Check for missing values
train_set.isnull().sum()
# To get idea about categorical variables
train_set.describe(include=['O'])
# To get idea about continuous variable
train_set.describe()
# to get idea about y variable
train_set['price_range'].value_counts()
train_set['touch_screen'].value_counts()
train_set['bluetooth'].value_counts()
train_set['dual_sim'].value_counts()
# To replace categorical variables
l = ['touch_screen','bluetooth','dual_sim','wifi','4g','3g']
for i in l:
    train_set[i] = train_set[i].replace({'yes':1,'no':0})
train_set['price_range'] = train_set['price_range'].replace({'very low':0,'low':1,'medium':2,'high':3})
# Creating new features
train_set['total_mem']=train_set['ram']+train_set['internal_memory']
train_set['diag']=np.sqrt(((train_set['screen_height']**2)+(train_set['screen_width']**2)))
features=['ram','battery_power','resolution_width','resolution_height','internal_memory',
'mobile_weight','screen_height','talk_time','screen_width','touch_screen']
X = train_set[features]
Y = train_set['price_range']
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
trainX, testX, trainY, testY =  train_test_split(X, Y, test_size = .3)
dt = SVC()
parameters={'C':[0.05],'kernel':['linear']}
clf = GridSearchCV(dt,parameters)
clf.fit(trainX,trainY)
preds = clf.predict(testX)
accuracy = accuracy_score(testY, preds)
precision = precision_score(testY, preds,average='micro')
recall = recall_score(testY, preds,average='micro')
print (accuracy,precision,recall)
print (classification_report(testY,preds))
test_set = pd.read_csv('../input/test.csv')
l = ['touch_screen','bluetooth','dual_sim','wifi','4g','3g']
for i in l:
    test_set[i] = test_set[i].replace({'yes':1,'no':0})
test_set['total_mem']=test_set['ram']+test_set['internal_memory']
test_set['diag']=np.sqrt(((test_set['screen_height']**2)+(test_set['screen_width']**2)))
preds = clf.predict(test_set[features])
preds=pd.Series(preds)
test_set['price_range'] = preds.replace({0:'very low',1:'low',2:'medium',3:'high'})
test_set['price_range'].value_counts()
