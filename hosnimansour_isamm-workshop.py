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
train ='../input/train.csv'

train_data = pd.read_csv(train)

train_data.head()
train_data=train_data.fillna(method='bfill')

train_data.head()
train_data.corr()
train_feature=['Pclass','Fare','Sex','Parch']

train_feature_data = train_data[train_feature]

train_feature_data.head()
train_feature_data['Sex'],_=pd.factorize(train_feature_data['Sex'])

train_feature_data.head()
target = train_data.Survived
from sklearn.tree import DecisionTreeRegressor



result = DecisionTreeRegressor(random_state=1)



result.fit(train_feature_data,target)
test_data = pd.read_csv('../input/test.csv')

test_data.head()
test_data=test_data.fillna(method='bfill')
x_test = test_data[train_feature]
x_test.head()
x_test['Sex'],_=pd.factorize(x_test['Sex'])

x_test['Fare']=np.int64(x_test['Fare'])
submit = pd.DataFrame({'PassengerID':test_data.PassengerId,'Survived':np.int64(result.predict(x_test))})

submit.to_csv('tree.csv',encoding='utf-8',index=False)
submit.head()