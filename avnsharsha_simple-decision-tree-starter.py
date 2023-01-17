import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
train_set = pd.read_csv('../input/train.csv')
train_set.shape
train_set.head()
train_set.columns
train_set.isnull().sum()
train_set.describe()
train_set.describe(include=['O'])
train_set['touch_screen'].value_counts()
train_set['bluetooth'].value_counts()
train_set['dual_sim'].value_counts()
train_set['wifi'].value_counts()
train_set['4g'].value_counts()
train_set['3g'].value_counts()
train_set['price_range'].value_counts()
#Lets convert these categorical values into numeric
l = ['touch_screen','bluetooth','dual_sim','wifi','4g','3g']
for i in l:
    train_set[i] = train_set[i].replace({'yes':1,'no':0})
train_set['price_range'] = train_set['price_range'].replace({'very low':0,'low':1,'medium':2,'high':3})
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
features = list(set(train_set.columns) - set(['Id','price_range']))
features
X = train_set[features]
Y = train_set['price_range']
trainX, testX, trainY, testY =  train_test_split(X, Y, test_size = .3)
dt = DecisionTreeClassifier()
model = dt.fit(trainX,trainY)
preds = model.predict(testX)
accuracy = accuracy_score(testY, preds)
precision = precision_score(testY, preds,average='micro')
recall = recall_score(testY, preds,average='micro')
print (accuracy,precision,recall)
print (classification_report(testY,preds))
test_set = pd.read_csv('../input/test.csv')
l = ['touch_screen','bluetooth','dual_sim','wifi','4g','3g']
for i in l:
    test_set[i] = test_set[i].replace({'yes':1,'no':0})
test_set['price_range'] = model.predict(test_set[features])
test_set['price_range'] = test_set['price_range'].replace({0:'very low',1:'low',2:'medium',3:'high'})
#test_set[['Id','price_range']].to_csv('1st_sub.csv',index=False)
