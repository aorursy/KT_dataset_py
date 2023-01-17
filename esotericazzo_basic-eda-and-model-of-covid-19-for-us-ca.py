# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')

df_test = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')

df_submit = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv')

df_train.head()
df_train.info()
print(df_train['Province/State'].unique())

print(df_train['Country/Region'].unique())
import matplotlib.pyplot as plt

plt.figure(figsize=(14,6))

plt.hist(df_train['ConfirmedCases'],bins=10,color='green')

plt.xlabel('Confirmed Cases')

plt.ylabel('Count')

plt.title('Count of Confirmed Cases')
import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))

plt.plot(df_train['Fatalities'])

plt.xlabel('Fatalities')

plt.ylabel('Count')

plt.title('Graph of Fatalities ')
df_train = df_train[['Date','ConfirmedCases','Fatalities']]

df_train.head()
plt.figure(figsize=(15,10))

sns.barplot(x=df_train['Date'] , y = df_train['ConfirmedCases'])

plt.xticks(rotation=90)

plt.figure(figsize=(15,10))

sns.barplot(x=df_train['Date'] , y = df_train['Fatalities'])

plt.xticks(rotation=90)
df_train_new = df_train.query('ConfirmedCases > 0')

df_train_new
plt.figure(figsize=(15,10))

#sns.barplot(x=df_train_new['Date'] , y = df_train_new['Fatalities'])

sns.barplot(x=df_train_new['Date'] , y = df_train_new['ConfirmedCases'])

plt.xticks(rotation=45)

plt.title('ConfirmedCases as per Date')
df_train['Date'] = pd.to_datetime(df_train['Date'])

df_train.insert(1,'Week',df_train['Date'].dt.week)

df_train.insert(2,'Day',df_train['Date'].dt.day)

df_train.insert(3,'DayofWeek',df_train['Date'].dt.dayofweek)

df_train.insert(4,'DayofYear',df_train['Date'].dt.dayofyear)
df_train
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import BayesianRidge 

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
X = df_train.drop(['Date', 'ConfirmedCases', 'Fatalities'], axis=1)

y = df_train[['ConfirmedCases', 'Fatalities']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
y_train.head()
#Function that predicts the scores of models.

def predict_scores(reg_alg):

    m = reg_alg()

    m.fit(X_train, y_train['ConfirmedCases'])

    y_pred = m.predict(X_test)

    m_r = r2_score(y_test['ConfirmedCases'], y_pred)

    sc_Cases.append(m_r)

    

    m.fit(X_train, y_train['Fatalities'])

    y_pred = m.predict(X_test)

    m_r2 = r2_score(y_test['Fatalities'], y_pred)

    sc_Fatalities.append(m_r2)





    

reg_models = [KNeighborsRegressor, LinearRegression, RandomForestRegressor, GradientBoostingRegressor, DecisionTreeRegressor,BayesianRidge]



sc_Cases = []

sc_Fatalities = []



for x in reg_models:

    predict_scores(x)
sc_Cases
sc_Fatalities
models = pd.DataFrame({

    'Model': ['KNeighborsRegressor', 'LinearRegression', 'RandomForestRegressor', 'GradientBoostingRegressor', 'DecisionTreeRegressor','BayesianRidge' ],

    'ConfirmedCase_r2': sc_Cases,

    'Fatalities_r2' : sc_Fatalities

})



models
df_test.head()
df_test.info()
df_test = df_test[['ForecastId', 'Date']]



df_test['Date'] = pd.to_datetime(df_test['Date'])

df_test.insert(1,'Week',df_test['Date'].dt.week)

df_test.insert(2,'Day',df_test['Date'].dt.day)

df_test.insert(3,'DayofWeek',df_test['Date'].dt.dayofweek)

df_test.insert(4,'DayofYear',df_test['Date'].dt.dayofyear)



df_test.head()
model1 = RandomForestRegressor()

model1.fit(X_train, y_train['ConfirmedCases'])



model2 = RandomForestRegressor()

model2.fit(X_train, y_train['Fatalities'])



df_test['ConfirmedCases'] = model1.predict(df_test.drop(['Date', 'ForecastId'], axis=1))

df_test['Fatalities'] = model2.predict(df_test.drop(['Date', 'ForecastId', 'ConfirmedCases'], axis=1))
import warnings

warnings.filterwarnings('ignore')

df_results = df_test[['ForecastId', 'ConfirmedCases', 'Fatalities']] 

df_results['ConfirmedCases'] = df_results['ConfirmedCases'].astype(int)

df_results['Fatalities'] = df_results['Fatalities'].astype(int)



df_results.head()
df_results.to_csv('submission.csv', index=False)