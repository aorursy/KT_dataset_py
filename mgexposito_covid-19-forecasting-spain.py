# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

data_test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")
data_train.head()
spain_train = data_train[data_train["Country_Region"]=="Spain"]

spain_test = data_test[data_test["Country_Region"]=="Spain"]
spain_train.tail()
plt.figure(figsize=(10,8))

plt.plot(spain_train["ConfirmedCases"])

plt.xlabel("Time")

plt.ylabel("Number of confirmed cases")
plt.figure(figsize=(10,8))

plt.plot(spain_train["Fatalities"])

plt.xlabel("Time")

plt.ylabel("Number of fatalities")
spain_train.drop('Province_State',axis=1,inplace=True)
spain_train.tail()
spain_test.tail()
encoder = LabelEncoder()

spain_train['Date'] = encoder.fit_transform(spain_train['Date'])

spain_train.tail()
# eliminamos las columnas que no son necesarias

spain_train.drop('Country_Region',axis=1,inplace=True)

spain_train.drop('Id',axis=1,inplace=True)

spain_train.drop('Fatalities',axis=1,inplace=True)

X = spain_train.drop(["ConfirmedCases"], axis=1)

Y = spain_train['ConfirmedCases']

x_train, x_test, y_train, y_test = train_test_split (X, Y, test_size = 0.25, random_state=1)

x_train.tail()
LR = LinearRegression()

LR.fit(x_train, y_train)

print('Average rate Linear Regression: ', round(LR.score(x_test, y_test) * 100, 2), '%')
ridge = Ridge()

ridge.fit(x_train, y_train)

print('Average rate Ridge: ', round(ridge.score(x_test, y_test) * 100, 2), '%')
rf = RandomForestClassifier(n_estimators=20, max_samples=0.8, random_state=1)

rf.fit(x_train, y_train)

print('Average rate Random Forest: ', round(rf.score(x_test, y_test) * 100, 2), '%')