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



# selecting columns with more than 800 non blank data

features=pd.DataFrame(train.notnull().sum())

features.columns=['count']

features=features[features['count']>800]

training=train[features.index]



values=pd.DataFrame(train[features.index].notnull().sum())

values.columns=['val']

missing=values[values['val']!=1460]



train_full=training.dropna()



# text columns binary encoding

bi_train=pd.get_dummies(train_full)

bi_test=pd.get_dummies(test)



# selecting common columns after encoding

common=[]

for i in bi_train.columns:

    if i in bi_test.columns:

        common.append(i)

y=bi_train['SalePrice']

X=bi_train[common]

test=bi_test[common]



from sklearn.model_selection import train_test_split

from sklearn.linear_model import Lasso

from sklearn.preprocessing import MinMaxScaler



# splitting the training data to train & test to evaluate the model

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size=0.2)



#rescaling columns values

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

X_all_scaled=scaler.transform(X)



from sklearn.model_selection import GridSearchCV



reg = Lasso(max_iter = 10000)

par= {'alpha':[200, 250, 300, 350, 400, 450, 500]}

linlasso= GridSearchCV(reg, param_grid = par)

linlasso.fit(X_train_scaled, y_train)



r2_train = linlasso.score(X_train_scaled, y_train)

r2_test = linlasso.score(X_test_scaled, y_test)

r2_all = linlasso.score(X_all_scaled, y)



# filling missing values in test data

from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)



test_scaled = scaler.transform(test)





testnew=imp.fit_transform(test_scaled)



testnew=pd.DataFrame(testnew)

ypred=linlasso.predict(testnew)

y_pred=pd.DataFrame(ypred)

y_pred.index+=1461

y_pred.columns=['SalePrice']

y_pred.index.rename('Id', inplace=True)

y_pred.to_csv('new6Aprtrial.csv')