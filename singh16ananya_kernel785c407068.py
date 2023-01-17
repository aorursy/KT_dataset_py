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
train=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")
train.head()
test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")
test.head()
submission=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')
import matplotlib.pyplot as plt

import seaborn as sns
print("Number of Country_Region: ", train['Country_Region'].nunique())

print("Dates are ranging from day", min(train['Date']), "to day", max(train['Date']), ", a total of", train['Date'].nunique(), 

      "days")

print("The countries that have Province/Region given are : ", train[train['Province_State'].isna()==False]['Country_Region'].

      unique())
train.columns
train['Province_State'].unique()
show_cum = train.groupby(by='Country_Region')[['ConfirmedCases','Fatalities']].max().reset_index()

plt.figure(figsize=(40,20))

sns.barplot(x='ConfirmedCases',y='Country_Region',data=show_cum[show_cum['ConfirmedCases'] != 0].

            sort_values(by='ConfirmedCases',ascending=False).head(30))
plt.figure(figsize=(20,10))

sns.barplot(x='Fatalities',y='Country_Region',data=show_cum[show_cum['Fatalities'] != 0].

            sort_values(by='Fatalities',ascending=False).head(30))
confirmed_total_dates = train.groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_dates = train.groupby(['Date']).agg({'Fatalities':['sum']})

total_dates = confirmed_total_dates.join(fatalities_total_dates)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))

total_dates.plot(ax=ax1)

ax1.set_title("Global confirmed cases", size=13)

ax1.set_ylabel("Total Number of cases", size=13)

ax1.set_xlabel("Date", size=13)

fatalities_total_dates.plot(ax=ax2, color='orange')

ax2.set_title("Global fatalities cases", size=13)

ax2.set_ylabel("Total Number of cases", size=13)

ax2.set_xlabel("Date", size=13)
X_train=train.drop(columns=['Id','ConfirmedCases','Fatalities','Date'])

y_train_cc=train.ConfirmedCases

y_train_ft=train.Fatalities
from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

impute=SimpleImputer(strategy='most_frequent')

X_train_1=impute.fit_transform(X_train)

X_train_2=OneHotEncoder().fit_transform(X_train_1)
X_test=test.drop(columns=['ForecastId','Date'])

X_test_1=impute.fit_transform(X_test)

X_test_2=OneHotEncoder().fit_transform(X_test_1)
from sklearn.ensemble import RandomForestRegressor

model_cc=RandomForestRegressor()

model_cc.fit(X_train_2, y_train_cc)

model_cc.score(X_train_2, y_train_cc)
y_pred_cc=model_cc.predict(X_test_2)
y_pred_cc
model_ft=RandomForestRegressor()

model_ft.fit(X_train_2,y_train_ft)

model_ft.score(X_train_2, y_train_ft)
y_pred_ft=model_ft.predict(X_test_2)
y_pred_ft
result=pd.DataFrame({'ForecastId':submission.ForecastId, 'ConfirmedCases':y_pred_cc, 'Fatalities':y_pred_ft})

result.to_csv('/kaggle/working/submission.csv',index=False)

data=pd.read_csv('/kaggle/working/submission.csv')

data.head()