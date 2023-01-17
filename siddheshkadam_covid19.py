import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sn
submission=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/submission.csv")

submission.head()
submission.tail()
train=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")

train.head()
test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv")

test.head()
corr=train.corr()

sn.heatmap(corr,vmax=1.,square=False)

corr=test.corr()

sn.heatmap(corr,vmax=1.,square=False)
sn.barplot(x="Date",y="Long",data=test)
sn.barplot(x="Country/Region",y="Long",data=test)
sn.barplot(x="Country/Region",y="Lat",data=test)
#sn.barplot(x="ForecastId",y="Lat",data=train)
test.plot(kind='box',subplots=True,layout=(3,3),sharex=False,sharey=False)

#in lat column their are some outliers 
train.hist()
test.hist()
plt.scatter(test['ForecastId'],test['Lat'])
plt.scatter(test['Lat'],test['Long'])
plt.scatter(test['Lat'][:1000],test['Country/Region'][:1000])
plt.scatter(test['ForecastId'][:1000],test['Country/Region'][:1000],test['Lat'][:1000],color='red')
plt.scatter(test['Lat'],test['ForecastId'],color='red')