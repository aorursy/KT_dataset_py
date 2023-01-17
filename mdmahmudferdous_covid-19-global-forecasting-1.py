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
train=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")

test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv")

submission=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/submission.csv")
X_test1=test[['ForecastId']]+16188
X1=train[['Id']]

y_con=train[['ConfirmedCases']]

y_fat=train[['Fatalities']]
from sklearn.preprocessing import PolynomialFeatures

poly=PolynomialFeatures(2)

X=poly.fit_transform(X1)

X_test=poly.fit_transform(X_test1)
from sklearn.linear_model import Ridge, Lasso, SGDRegressor

model_con=Lasso()

model_con.fit(X, y_con);
y_pred_con=model_con.predict(X_test)

y_pred_con
model_fat=Lasso()

model_fat.fit(X, y_fat)
y_pred_fat=model_fat.predict(X_test)

y_pred_fat
y_pred_con1=y_pred_con.ravel()

y_pred_fat1=y_pred_fat.ravel()
data={'ForecastId':submission.ForecastId,'ConfirmedCases':y_pred_con1, 'Fatalities':y_pred_fat1}

result=pd.DataFrame(data, index=submission.index)

result.to_csv('/kaggle/working/submission.csv', index=False)

m1=pd.read_csv('/kaggle/working/submission.csv')

m1.head()
import matplotlib.pyplot as plt

%matplotlib inline
hor=train['Id']

ver=train['ConfirmedCases']

plt.plot(hor, ver)

plt.show()

hor=m1['ForecastId']

ver=m1['ConfirmedCases']

plt.plot(hor, ver)

plt.show()
hor=train['Id']

ver=train['Fatalities']

plt.plot(hor, ver)

plt.show()

hor=m1['ForecastId']

ver=m1['Fatalities']

plt.plot(hor, ver)

plt.show()
train['Country/Region'].value_counts()
train.tail()
top_risk=train[train.Date=='2020-03-18'].groupby('Country/Region').sum().sort_values(by='ConfirmedCases',ascending=False).head(10)

top_risk
