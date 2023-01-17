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

import seaborn as sb

import matplotlib.pyplot as plt

import time

from datetime import datetime

# Any results you write to the current directory are saved as output.
covid_train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

covid_train.head()
covid_test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

covid_test.head()
covid_train.shape
covid_test.shape
covid_train.describe()
covid_train.info()
covid_train['Date']=pd.to_datetime(covid_train['Date'],infer_datetime_format=True)

covid_test['Date']=pd.to_datetime(covid_test['Date'],infer_datetime_format=True)
covid_train.info()
covid_test.info()
covid_train.hist()
covid_test.hist()


covid_train.shape
corr=covid_train.corr()

sb.heatmap(corr,vmax=1.,square=True)
g=sb.heatmap(covid_train[["Id","ConfirmedCases","Fatalities"]].corr(),annot=True,fmt=".2f",cmap="coolwarm")
covid_x=pd.DataFrame(covid_train.iloc[:,-1])

covid_x.head()
covid_y=pd.DataFrame(covid_train.iloc[:,-2])

covid_y.head()
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(covid_x,covid_y,test_size=0.3)

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

regression=LinearRegression()

regression.fit(X_train,Y_train)

#regression['Date']=regression['Date'].astype(int)

tree_regressor=DecisionTreeRegressor()

tree_regressor.fit(X_train,Y_train)



y_pred_lin=regression.predict(X_test)

y_pred_df=pd.DataFrame(y_pred_lin,columns=['Predict'])

Y_test.head()
y_pred_df.head()
y_pred_tree=tree_regressor.predict(X_test)

y_tree_pred_df=pd.DataFrame(y_pred_tree,columns=['Predict_tree'])

y_tree_pred_df.head()
plt.figure(figsize=(5,5))

plt.title('Actual vs Prediction')

plt.xlabel('Fatalities')

plt.ylabel('Predicted')

plt.legend()

plt.scatter((X_test['Fatalities']),(Y_test['ConfirmedCases']),c='red')

plt.scatter((X_test['Fatalities']),(y_pred_df['Predict']),c='cyan')

plt.show()

            
sub=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')

sub.to_csv('submission_csv',index=False)