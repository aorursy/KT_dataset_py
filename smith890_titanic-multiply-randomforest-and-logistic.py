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
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np



test_data=pd.read_csv('../input/titanic/test.csv')

train_data=pd.read_csv('../input/titanic/train.csv')

train_data.set_index(['PassengerId'],inplace=True)

test_data.set_index(['PassengerId'],inplace=True)

full_data=train_data.append(test_data)

train_y=train_data.Survived

train_data.info()
train_data.head()

cols=train_data.corr()['Survived'].index

cols
#throw out feature:Name ,Survived

full_data=full_data.drop(columns=['Name','Survived'])

#Convert categorical variable into dummy/indicator variables.

full_data_dummy=full_data.copy()

full_data_dummy=pd.get_dummies(full_data,columns=['Pclass','Embarked','Sex','Ticket'])

full_data_dummy.head()
#throw out Cabin which has too much Nan

full_data_dummy=full_data_dummy.drop('Cabin',axis=1)
full_data_dummy.isnull().sum().sort_values(ascending=False).head(15)
#fill Nan value

full_data_dummy=full_data_dummy.fillna(full_data_dummy.mean())
#Normalization

numeric_cols=['Age','Fare']



numeric_col_mean=full_data_dummy.loc[:,numeric_cols].mean()

numeric_col_std=full_data_dummy.loc[:,numeric_cols].std()

full_data_dummy.loc[:,numeric_cols]=(full_data_dummy.loc[:,numeric_cols]-numeric_col_mean)/numeric_col_st
#show that is there has unique values 

plt.scatter(full_data.fillna(full_data.mean()).Age,full_data.Fare)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor
train_data_dummy=full_data_dummy.loc[train_data.index]

test_data_dummy=full_data_dummy.loc[test_data.index]
#find best parameter

N_estimators=[20,50,100,150,200,250,300]

test_scores=[]

train_X=train_data_dummy

for N in N_estimators:

    clf=RandomForestRegressor(n_estimators=N,max_features=0.3)

    test_score=np.sqrt(-cross_val_score(clf,train_X,train_y,cv=5,scoring='neg_mean_squared_error'))

    test_scores.append(np.mean(test_score))  

plt.plot(N_estimators,test_scores)

plt.title('N_estimator vs CV Error')
rf=RandomForestRegressor(n_estimators=200,max_features=0.3)

rf.fit(train_X,train_y)
lg=LogisticRegression(C=1.0)

lg.fit(train_X,train_y)
#mix predictions

rf_predict=rf.predict(test_data_dummy)

lg_predict=lg.predict_proba(test_data_dummy)[:,1]

y_final=(rf_predict+lg_predict)/2

y_final=y_final.round()

y_final=y_final.astype(int)

y_final
submission=pd.DataFrame(data={'PassengerId':test_data.index,'Survived':y_final})

submission.to_csv('titanic_subm.csv',index=False)