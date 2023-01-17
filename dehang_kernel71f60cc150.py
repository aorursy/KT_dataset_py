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
from sklearn.linear_model import LogisticRegression,Lasso,Ridge,LinearRegression

from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV

from sklearn.metrics import r2_score

from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

%matplotlib inline

warnings.filterwarnings('ignore')
data=pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')
data.shape
data.head()
data.columns
data.info()
data.describe()
data.isnull().sum()
data.isnull().sum()/len(data)*100
data.drop('company',axis=1,inplace=True)
data.is_canceled.value_counts()/len(data['is_canceled'])
plt.pie(data['is_canceled'].value_counts(),autopct='%.2f%%',explode=[0,0.08],shadow=True,labels=['No Canceled','Canceled'])
data['country'].fillna(data['country'].mode()[0],inplace=True)
data['agent'].value_counts()
data['agent'].fillna(data['agent'].notnull().median(),inplace=True)
data['children'].fillna(data['children'].notnull().mean(),inplace=True)
data['hotel'].unique()
obj_columns=data.select_dtypes(include=['object'])

columns=obj_columns.columns
columns
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

for i in range(0,len(columns)):

    data[columns[i]]=le.fit_transform(data[columns[i]])
data.head()
data.hotel.value_counts()
cor=data.corr()

plt.figure(figsize=(24,12))

sns.heatmap(cor,cmap='YlGnBu',vmin=0,vmax=1,annot=True)
help(sns.heatmap)
regressions={'Logistic Regression':LogisticRegression(solver='lbfgs'),

           'Linear Regression':LinearRegression(),

           'Ridge Regression':Ridge(alpha=1.0),

           'Lasso Regression':Lasso(alpha=0.1),

           'Random Forest Regression':RandomForestRegressor(n_estimators = 100, random_state = 42),

           'Adaboost Regression':AdaBoostRegressor(n_estimators = 100, random_state = 42)}
X=data.drop('previous_cancellations',axis=1)

y=data['previous_cancellations']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=111)
for regression in regressions:

    reg=regressions[regression].fit(X_train,y_train)

    y_pred=reg.predict(X_test)

    print('---'*10,regression,'---'*10)

    print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))

    print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_pred))

    print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

    print('r2_score:',r2_score(y_test,y_pred))

    print('---'*25)
rf=RandomForestRegressor(max_features='auto',n_jobs=-1,oob_score=True,random_state=111)

params={'n_estimators':[50,100,150]

       }

gs=GridSearchCV(rf,param_grid=params,cv=5,verbose=0)
gs.fit(X_train,y_train)
gs.best_estimator_
gs.best_params_
gs.best_score_