# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

import sympy as sp



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
submission=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')

submission.head()
test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')

test.head()
train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

train.head()
train.info()
submission.info()
test.info()
train['ConfirmedCases'].value_counts()
train['Fatalities'].value_counts()
x=train.groupby('Country/Region')['Country/Region','ConfirmedCases','Fatalities'].mean()

x

train.iloc[700:4600,1:9]

sns.regplot(train['Fatalities'],train['ConfirmedCases'])

plt.ylim(0,)
train.mean()
f=(train['ConfirmedCases']-train['ConfirmedCases'].mean())/train['ConfirmedCases'].std()

print(f)

f.plot()



v=(train['Fatalities']-train['Fatalities'].mean())/train['Fatalities'].std()

print(v)

v.plot()
x.plot.pie(y='ConfirmedCases',figsize=(10,20))

x.plot.line(subplots=True)
x.plot.kde(bw_method=0.3,subplots=True)
x.plot.hist(bins=12,alpha=0.5,subplots=True)
x.plot.bar(figsize=(35,35))