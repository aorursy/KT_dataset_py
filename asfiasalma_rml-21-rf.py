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
data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

data.head()
data.info()
data['Confirmed'] = data['Confirmed'].astype("int32")

print(data['Confirmed'].sum())

print(data['Recovered'].sum()/data['Confirmed'].sum() * 100)

print(data['Deaths'].sum()/data['Confirmed'].sum() * 100)



X = data['Confirmed'].groupby(data["ObservationDate"]).sum()

dates = X.index

X[dates[0]]

len(dates)

confirmed = []

for i in range(len(dates)):

    confirmed.append(X[dates[i]])



confirmed
days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)

days_in_future = 5

future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)

adjusted_dates = future_forcast[:-5]

import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize=(20, 12))

plt.plot(days_since_1_22, confirmed)

plt.title('# of Coronavirus Cases Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=15)

plt.show()
days_since_1_22 = np.array(days_since_1_22).reshape(-1, 1)

confirmed = np.array(confirmed).reshape(-1, 1)
from sklearn.model_selection import train_test_split 

X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, confirmed, test_size=0.25, random_state=101) 
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

# model = RandomForestRegressor(n_estimators=10, random_state=0)

model = LinearRegression()

model.fit(X_train_confirmed,y_train_confirmed)

print('Train Accuracy:', r2_score(y_train_confirmed, model.predict(X_train_confirmed)))

y_pred = model.predict(X_test_confirmed)

print('Test Accuracy:', r2_score(y_test_confirmed, y_pred))
# rf_pred = model.predict(future_forcast)

linear_pred = model.predict(future_forcast)
plt.figure(figsize=(20, 12))

plt.plot(days_since_1_22, confirmed)

plt.plot(future_forcast, linear_pred, linestyle='dashed', color='purple')

plt.title('# of Coronavirus Cases Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.legend(['Confirmed Cases', 'Linear predictions'])

plt.xticks(size=15)

plt.show()