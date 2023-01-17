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

#importing the required libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import Pool,CatBoostClassifier,cv
#reading training and testing data

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.info()
train_df.drop(['Name','Ticket'],axis = 1,inplace =True)
train_df['Age'].mean()
train_df['Age'].fillna(30,inplace =True)
train_df.isnull().sum()
train_df['Cabin'].fillna('NA',inplace = True)
train_df.dropna(inplace =True)
train_df.head()
x = train_df.drop('Survived',axis = 1)
y = train_df['Survived']
cat_features_index = np.where(x.dtypes != float)[0]
xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size=.90,random_state=1234)
model = CatBoostClassifier(eval_metric='Accuracy',use_best_model=True,random_seed=42)
model.fit(xtrain,ytrain,cat_features=cat_features_index,eval_set=(xtest,ytest))
test_df['Age'].fillna(30,inplace = True)
test_df['Cabin'].fillna('NA',inplace = True)
test_df['Fare'].mean()
test_df.fillna(35.62,inplace = True)
submission = test_df.drop(['Name','Ticket'],axis = 1)
pred = model.predict(submission)
submission['Survived'] = pred
submission.head()
sub = submission.drop(['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked'],axis = 1)
sub.to_csv('catboost.csv',index = False)
