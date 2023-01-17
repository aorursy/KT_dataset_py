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






submission=pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv')

submission.head()
submission.info()
x=submission.groupby('ForecastId')['ForecastId','ConfirmedCases','Fatalities'].mean()

x.plot(figsize=(5,5),subplots=True)
df=submission[['ConfirmedCases','Fatalities','ForecastId']]

x_data=df['ConfirmedCases']

y_data=df['Fatalities']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,random_state=0,test_size=0.1)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt

lm=LinearRegression()

y=StandardScaler()

x=submission['ConfirmedCases'].values.reshape(-1,1)

y=submission['Fatalities'].values.reshape(-1,1)

lm.fit(x,y)

s=lm.predict(x)

plt.figure()

plt.scatter(submission['ConfirmedCases'],submission['Fatalities'],c='black')

plt.plot(submission['ConfirmedCases'],s,c='blue',linewidth=2)

plt.show()
lm.coef_
lm.intercept_
train.head()
s=train.drop(['Lat','Long'],axis=1)

s
s.describe()
from scipy import stats

d,f=stats.pearsonr(s['ConfirmedCases'],s['Fatalities'])

d,f
p=cross_val_score(lm,x,y,cv=3)

p
s.plot.line(figsize=(5,5))
import seaborn as sns

sns.regplot(s['ConfirmedCases'],s['Fatalities'])
s.plot.bar(figsize=(10,10))
s.plot.kde(figsize=(7,7))
a=s.groupby('Province/State')['Province/State','ConfirmedCases','Fatalities'].mean()

a
s['ConfirmedCases'].sum()
submission.to_csv('submission.csv')