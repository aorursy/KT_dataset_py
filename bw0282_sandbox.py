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
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv("../input/train.csv")

train.head()
train.info()

sns.distplot(train['target'])
# std is higher then mean
train['target'].describe()
#target 값 범쥐 존재
train.sort_values('target',ascending=False)
len(train['target'].unique())
target_val = train.groupby('target').count()['ID']
target_val = target_val.to_frame()
target_val.columns = ['count']
target_val = target_val.sort_values('count',ascending=False)
target_val[target_val['count'] ==1]['count'].sum()
target_val['count'].sum()
target_val.head()
first = train[train['target'] == 2000000.0]
sample = first[first['48df886f9'] > 0]
sam_col = sample.columns
sample['48df886f9'][2303] >0
sam_col = sample.columns.drop(['ID','target'])
sam_list = []
for i in sam_col:
    if int(sample[i][2303]) > 0:
        sam_list.append(i)






sam_ = sample[sam_list]
int(sam_.mean(axis=1))
sample['target']
1186339/2000000
sample[sam_list]
train[['target','20aa07010']].corr()
train
test = pd.read_csv('../input/test.csv')
test.info()
train.info()

x_train = train.drop(['ID','target'],axis=1)
x_test = test.drop(['ID'],axis=1)
colsToRemove = []
for col in x_train.columns:
    if x_train[col].std() == 0:
        colsToRemove.append(col)
# # remove constant columns in the training set
# x_train.drop(colsToRemove, axis=1, inplace=True)

# remove constant columns in the test set
x_test.drop(colsToRemove, axis=1, inplace=True) 

print("Removed `{}` Constant Columns\n".format(len(colsToRemove)))
print(colsToRemove)
# Check and remove duplicate columns
colsToRemove = []
colsScaned = []
dupList = {}

columns = x_train.columns

for i in range(len(columns)-1):
    v = x_train[columns[i]].values
    dupCols = []
    for j in range(i+1,len(columns)):
        if np.array_equal(v, x_train[columns[j]].values):
            colsToRemove.append(columns[j])
            if columns[j] not in colsScaned:
                dupCols.append(columns[j]) 
                colsScaned.append(columns[j])
                dupList[columns[i]] = dupCols
                
# remove duplicate columns in the training set
x_train.drop(colsToRemove, axis=1, inplace=True) 

# remove duplicate columns in the testing set
x_test.drop(colsToRemove, axis=1, inplace=True)

print("Removed `{}` Duplicate Columns\n".format(len(dupList)))
print(dupList)
print("train",x_train.shape)
print("test", x_test.shape)
from sklearn.ensemble import RandomForestRegressor
y_train = train['target']
cls = RandomForestClassifier(random_state=42)
cls.fit(x_train,y_train)
y_train = np.log1p(train["target"].values)
y_train = y_train.astype('int')
clf.fit(x_train,y_train)
feat_importances_rf = pd.Series(clf.feature_importances_, index = x_train.columns)
feat_importances_rf = feat_importances_rf.nlargest(25)
plt.figure(figsize=(16,8))
feat_importances_rf.plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()
from sklearn.linear_model import Lasso

fea = feat_importances_rf.to_frame()
col = list(fea.index)
x_train = x_train[col]
rlf=RandomForestRegressor(max_depth=5,n_jobs=-1)
from sklearn.ensemble import RandomForestRegressor
rlf =RandomForestRegressor()
rlf.fit(x_train,y_train)
x_test = x_test[col]
pred = rlf.predict(x_test)
pred= np.exp2(pred)
sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = pred
sub.to_csv('scrach.csv')
