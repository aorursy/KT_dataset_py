# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')
df.head()
df_date=df.groupby(['date']).sum()
df_date=df_date.drop(columns=['fips'])
import seaborn as sns

fig = plt.figure(figsize=(24,8))
ax = fig.add_subplot(1, 1, 1)

ax.plot(df_date.index, df_date['cases'],color='black')
ax.plot(df_date.index, df_date['deaths'],color='red')

ax.set_xticks(df_date.index)
ax.set_xticklabels(df_date.index, rotation=75)

plt.xlabel('date',fontsize=15)
plt.ylabel('accum_patients',fontsize=15)
plt.legend(['cases','deaths'],loc='best')

from matplotlib import ticker
ax.yaxis.set_major_locator(ticker.MultipleLocator(20000))#xaxis、yaxisを使い分けて利用する

plt.show()
rate=round(df_date['deaths']/df_date['cases'],2)
fig = plt.figure(figsize=(24,8))
ax = fig.add_subplot(1, 1, 1)

ax.plot(rate.index, rate.values,color='black')

ax.set_xticks(rate.index)
ax.set_yticks(np.linspace(0, 0.1, 11))
ax.set_xticklabels(rate.index, rotation=75)

plt.xlabel('date',fontsize=15)
plt.ylabel('death_rate',fontsize=15)

#from matplotlib import ticker
#ax.yaxis.set_major_locator(ticker.MultipleLocator(20000))#xaxis、yaxisを使い分けて利用する

plt.show()
pre_date=df_date.reset_index()
pre_date['date']=pre_date['date'].astype('datetime64')
pre_date=pre_date.drop(columns=['deaths']).rename(columns={"cases":"y","date":"ds"})

from fbprophet import Prophet

m=Prophet()
m.fit(pre_date.tail(10))
future=m.make_future_dataframe(periods=10)
forecast=m.predict(future)
forecast

figure = m.plot(forecast,xlabel='Date',ylabel='cases')
pre_date=df_date.reset_index()
pre_date['date']=pre_date['date'].astype('datetime64')
pre_deaths=pre_date.drop(columns=['cases']).rename(columns={"deaths":"y","date":"ds"})

from fbprophet import Prophet

m=Prophet()
m.fit(pre_deaths.tail(8))
future=m.make_future_dataframe(periods=10)
forecast=m.predict(future)
forecast

figure = m.plot(forecast,xlabel='Date',ylabel='deaths')
state=df.groupby(['state']).sum()
cases=state.drop(columns=['fips']).sort_values('cases')
deaths=state.drop(columns=['fips']).sort_values('deaths')
fig = plt.figure(figsize=(12,12))

ax = fig.add_subplot(1, 2, 1)
ax.barh(cases.index, cases['cases'])
plt.xlabel('cases',fontsize=15)


ax = fig.add_subplot(1, 2, 2)
ax.barh(deaths.index, deaths['deaths'],color='red')
plt.xlabel('deaths',fontsize=15)


#ax.set_xticklabels(state['cases'], rotation=75)



#from matplotlib import ticker
#ax.xaxis.set_major_locator(ticker.MultipleLocator(1))#xaxis、yaxisを使い分けて利用する

plt.show()