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
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

gender=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

print(len(test.index),len(gender.index))
#drop the Nan

train.dropna(inplace=True)

#divide the train dataset into train_data and train target

train_data=train[['Sex','Age','Fare']]

train_target=train['Survived']

#let the sex get_dummies

Sex_dummies=pd.get_dummies(train_data['Sex'])

finally_train_data=pd.concat([train_data,Sex_dummies],join='inner',axis=1)

finally_train_data.drop('Sex',axis=1,inplace=True)

#handle the test_data

test=pd.concat([test,gender],axis=1)

#print('缺失值统计：',test.isnull().sum)

#print('去空值之前的数据为：',len(test.index))

test.dropna(inplace=True)

#print('去空值后为：',len(test.index))

test_data=test[['Sex','Age','Fare']]

test_target=test['Survived']

print('去空值后：',len(test_data.index))

#let the sex get_dummies

test_Sex_dummies=pd.get_dummies(test_data['Sex'])

finally_test_data=pd.concat([test_data,test_Sex_dummies],join='inner',axis=1)

finally_test_data.drop('Sex',axis=1,inplace=True)

print(len(finally_test_data.index))

#train the model with svm

from sklearn.svm import SVC

svm=SVC().fit(finally_train_data,train_target)

target_predict=svm.predict(finally_test_data)

#judge the effictiveness of svm model

right_predict=np.sum(target_predict==test_target)

flase_predict=test_target.size-right_predict

percent_right_predict=right_predict/test_target.size

print(right_predict,flase_predict,percent_right_predict)