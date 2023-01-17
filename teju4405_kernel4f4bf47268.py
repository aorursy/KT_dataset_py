# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
submission=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')

submission.head()
train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

train.head()

submission.info()
train.info()
s=submission.groupby('ForecastId')['ForecastId','ConfirmedCases','Fatalities'].mean()

s
x=train.groupby('Country/Region')['Country/Region','ConfirmedCases','Fatalities'].mean()

x
s.plot(figsize=(5,5))
x.plot(figsize=(20,20))
x.plot.pie(figsize=(20,30),subplots=True)
x.plot.kde(bw_method=0.3,subplots=True)
x.plot.bar(figsize=(35,35))
train.mean()
train['ConfirmedCases'].value_counts()
train['Fatalities'].value_counts()
y=(train['ConfirmedCases']-train['ConfirmedCases'].mean())/train['ConfirmedCases'].std()

print(y)

y.plot()
T=(train['Fatalities']-train['Fatalities'].mean())/train['Fatalities'].std()

print(T)

T.plot()
import seaborn as sns



sns.regplot(train['ConfirmedCases'],train['Fatalities'])

plt.ylim(0,)
x.plot.box(figsize=(10,10))
train.describe()
train['ConfirmedCases'].sum()
train['Fatalities'].sum()
train.to_csv('submission.csv',index=False)