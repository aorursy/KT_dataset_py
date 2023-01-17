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
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

print(train.info())

print(train.isnull().any().describe())#To check if there are any missing data in the training set.
#The output confirms that there are no missing data in the training set.

x_train=train.drop(['target','id'],axis=1)

y_train=train['target']

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

x_train=scaler.fit_transform(x_train)

from sklearn.linear_model import LogisticRegression

model=LogisticRegression(class_weight='balanced', solver='liblinear', penalty ='l1', C= 0.1, max_iter=10000)

model.fit(x_train,y_train)

print("Score on training data: " + str(model.score(x_train,y_train)*100) + "%")
x_test=test.drop('id',axis=1)

x_test=scaler.fit_transform(x_test)

y_test=model.predict_proba(x_test)

print(y_test)

ans=pd.DataFrame({'id':test['id'],'target':y_test[:,1]})

ans.to_csv('output.csv',index=False)