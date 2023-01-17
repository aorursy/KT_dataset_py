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

from sklearn.ensemble import GradientBoostingRegressor



# splitting the training data to train & test to evaluate the model performance,

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size=0.2)





from sklearn.model_selection import GridSearchCV



regr = GradientBoostingRegressor()

par={'learning_rate':[0.001, 0.1,0.2,0.3,1,10],'max_depth':[2,3,4,5,6], 'n_estimators':[80,90,100,110,120]}

grd=GridSearchCV(regr, param_grid=par)

grd.fit(X_train, y_train)



grd.score(X_test, y_test)



# filling missing values in test data

from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

testnew=imp.fit_transform(test)

testnew=pd.DataFrame(testnew)

ypred=grd.predict(testnew)

y_pred=pd.DataFrame(ypred)

y_pred.index+=1461

y_pred.columns=['SalePrice']

y_pred.index.rename('Id', inplace=True)

y_pred.to_csv('10-7Aprtrial.csv')