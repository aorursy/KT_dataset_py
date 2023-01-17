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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
sample = pd.read_csv('../input/sample_submission.csv')

sample.shape
sample.head(10)
train_data.shape # 371,7
cols = train_data.columns
train_data.head(10)
print(test_data.shape) # 259, 6
test_data.head(5)
Y = train_data['type']
train_data = train_data.drop(['id'], axis=1)
train_data = train_data.drop(['type'], axis=1)
test_data = test_data.drop(['id'], axis=1)
print(train_data.columns)
print(test_data.columns)
def exploreData(data, column):
    return data[column].value_counts()

def imputation(data, column, value):
    data.loc[data[column].isnull(), column] = value
    
def countMissing(data):
   missing = data.columns[data.isnull().any()].tolist()
   return missing
print(train_data.info())
print('train missing:', countMissing(train_data))
print(test_data.info())
print('test missing:', countMissing(test_data))
'''
cols = train_data
for col in cols:
    #print(exploreData(train_data, cols[4]))
    print(exploreData(train_data, col))
'''    
print(exploreData(train_data, 'color'))
print(exploreData(test_data, 'color'))
cat_cols = ['color']

for col in cat_cols:
    train_data = pd.concat((train_data, pd.get_dummies(train_data[col], prefix=col)), axis=1)
    del train_data[col]
    test_data = pd.concat((test_data, pd.get_dummies(test_data[col], prefix=col)), axis=1)
    del test_data[col]
print(train_data.columns)
print(test_data.columns)
# min-max scaling
train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())
test_data = (test_data - test_data.min()) / (test_data.max() - test_data.min())
train_data.head(5)
print(Y.value_counts())
# change to numerical before feeding into the model; otherwise error will be encountered. 
Y = [0 if y == 'Ghoul' else 1 if y == 'Goblin' else 2 for y in Y]
# get features importance scores from random forest
from sklearn.ensemble import RandomForestRegressor
rfe = RandomForestRegressor(n_estimators = 500)
rfe.fit(train_data, Y)
imp_score = rfe.feature_importances_
imp_score = pd.DataFrame({'feature': train_data.columns, 'score': imp_score})
print(imp_score.sort_values(by = 'score', ascending = False))
# cross validatiion
neighbors = [1, 2, 5, 10]
param = []
scores = []

start = datetime.now()
print('start time: ', start)
for n in neighbors:
    knn = KNeighborsClassifier(n_neighbors = n)
    scores.append(cross_val_score(knn, train_data, Y, scoring="accuracy", cv = 10).mean())
    param.append(n)

finish = datetime.now()
secs = (finish-start).seconds
print('elapsed time in minutes: ', secs/60.0)
scores = pd.DataFrame({'parameter': param, 'score': scores})
print(scores.sort_values(by = 'score', ascending = False))

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(train_data, Y)
res = knn.predict(test_data)
#Y = [0 if y == 'Ghoul' else 1 if y == 'Goblin' else 2 for y in Y]
type = ['Ghoul' if r == 0 else 'Goblin' if r == 1 else 'Ghost' for r in res]
type = pd.Series(type)
print(type.value_counts())
sample = pd.read_csv('../input/sample_submission.csv')
id = sample['id']

df = pd.DataFrame({'type': type})
df['id'] = id
#df.columns = ['ImageId', 'Label']
df = df[['id', 'type']]
df.columns
print(df.head(5))
df['type'].value_counts()
df.to_csv('knn_submission.csv', index=False)
print(os.listdir('.'))
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from datetime import datetime

alphas = [0.01, 0.03, 0.1, 0.3, 1.0, 10]
regs = ["l1","l2"]
scores_1 = []
param = []
scores_2 = []

start = datetime.now()
print('start time: ', start)
for alpha in alphas:
   for reg in regs:
       lm1 = OneVsOneClassifier(linear_model.LogisticRegression(C = alpha, multi_class = 'multinomial', solver = 'saga', penalty = reg))
       lm2 = OneVsRestClassifier(linear_model.LogisticRegression(C = alpha, multi_class = 'multinomial', solver = 'saga', penalty = reg))
       scores_1.append(cross_val_score(lm1, train_data, Y, scoring="accuracy", cv = 10).mean())
       scores_2.append(cross_val_score(lm2, train_data, Y, scoring="accuracy", cv = 10).mean())
       param.append([alpha, reg])

finish = datetime.now()
secs = (finish-start).seconds
print('elapsed time in minutes: ', secs/60.0)
scores_1 = pd.DataFrame({'parameter': param, 'score': scores_1})
scores_2 = pd.DataFrame({'parameter': param, 'score': scores_2})
print('OneVsONe')
print(scores_1.sort_values(by = 'score', ascending = False))
print('OneVsRest')
print(scores_2.sort_values(by = 'score', ascending = False))

#OneVsOne
lm1 = OneVsOneClassifier(linear_model.LogisticRegression(C = 10, multi_class = 'multinomial', solver = 'saga', penalty = 'l1'))
lm1.fit(train_data, Y)
res = lm1.predict(test_data)
print(res)
type = ['Ghoul' if r == 0 else 'Goblin' if r == 1 else 'Ghost' for r in res]
type = pd.Series(type)
print(type.value_counts())
#convert result to df
df = pd.DataFrame({'type': type})
df['id'] = id
#df.columns = ['ImageId', 'Label']
df = df[['id', 'type']]
df.columns
print(df.head(5))
df['type'].value_counts()
#output ressult to csv file
df.to_csv('onevsone_submission.csv', index=False)
print(os.listdir('.'))
#OneVsRest
lm2 = OneVsRestClassifier(linear_model.LogisticRegression(C = 1.0, multi_class = 'multinomial', solver = 'saga', penalty = 'l1'))
lm2.fit(train_data, Y)
res = lm2.predict(test_data)
print(res)
type = ['Ghoul' if r == 0 else 'Goblin' if r == 1 else 'Ghost' for r in res]
type = pd.Series(type)
print(type.value_counts())
#convert result to df
df = pd.DataFrame({'type': type})
df['id'] = id
#df.columns = ['ImageId', 'Label']
df = df[['id', 'type']]
df.columns
print(df.head(5))
df['type'].value_counts()
#output ressult to csv file
df.to_csv('onevsrest_submission.csv', index=False)
print(os.listdir('.'))