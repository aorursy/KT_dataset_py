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
#Based on https://www.kaggle.com/mielek/covid19-forecasting-xgboost and some creativity of mine...
train=pd.read_csv(r'/kaggle/input/covid19-global-forecasting-week-4/train.csv')
test=pd.read_csv(r'/kaggle/input/covid19-global-forecasting-week-4/test.csv')
train.head()
train['Date']=pd.to_datetime(train.Date)
test['Date']=pd.to_datetime(test.Date)
train.loc[:, 'Date'] = train.Date.dt.strftime('%y%m%d')
train.loc[:, 'Date'] = train['Date'].astype(int)

test.loc[:, 'Date'] = test.Date.dt.strftime('%y%m%d')
test.loc[:, 'Date'] = test['Date'].astype(int)
train.drop('Province_State',axis=1, inplace=True)
test.drop('Province_State',axis=1, inplace=True)
# have to predict confirmed cases and Fatalities
train.info()
#before that a bit of plot to check no. of confirmed cases
train['Country_Region'].value_counts().iloc[:10].plot(kind='bar')
import seaborn as sns
sns.scatterplot(x=train["ConfirmedCases"], y=train["Fatalities"])
#have to change country to numeric
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
#train.Country_Region=le.fit_transform(train.Country_Region)
#test.Country_Region=le.fit_transform(test.Country_Region)
train['Country_Region']=le.fit_transform(train['Country_Region'])
test['Country_Region']=le.fit_transform(test['Country_Region'])
train['Country_Region'].value_counts()

#have already checked no null values... 
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
train['Country_Region']
#The single bracket with output a Pandas Series, while a double bracket will output a Pandas DataFrame
X=train[['Country_Region','Date']]
y=train[['ConfirmedCases']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
X_train.head()
#so now.. lets create our base model
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
base_model=RandomForestRegressor()

base_model.fit(X_train,y_train)
y_pred=base_model.predict(X_test)
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test,y_pred))
print(y_pred)
next_model=XGBRegressor(n_estimators=10000)
next_model.fit(X_train,y_train)
y_pred_next=next_model.predict(X_test)
print(y_pred_next)
print(mean_absolute_error(y_test,y_pred_next))
test.head()
testid=test.ForecastId
#for final submission
test.drop('ForecastId', axis=1, inplace=True)
X=train[['Country_Region','Date']]
y=train[['ConfirmedCases']]
base_model.fit(X,y)
y_pred_confirm=base_model.predict(test) # as based model has already been trained with confirmed cases , but still trained one more time before predict
X=train[['Country_Region','Date']]
y=train[['Fatalities']]
base_model.fit(X,y)
y_pred_fat=base_model.predict(test)
print(y_pred_confirm)
print(y_pred_fat)
df_sub = pd.DataFrame()
df_sub['ForecastId'] = testid
df_sub['ConfirmedCases'] = y_pred_confirm
df_sub['Fatalities'] = y_pred_fat
df_sub.to_csv('submission.csv', index=False)
df_sub
