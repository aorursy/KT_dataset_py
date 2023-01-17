import numpy as np             

import pandas as pd            

import matplotlib.pylab as plt

from datetime import datetime

%matplotlib inline             

df_crude_oil = pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/Crude_oil_trend_From1986-10-16_To2020-03-31.csv', header=0, index_col=0, parse_dates=True, squeeze=False)

df_train = pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/COVID-19_train.csv', header=0, index_col=0, parse_dates=True, squeeze=False)

df_test = pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/COVID-19_test.csv', header=0, index_col=0, parse_dates=True, squeeze=False)



#  To check the shape of the dataset 

print('The shape of crude oil price data:',df_crude_oil.shape)

print('The shape of train data :',df_train.shape)

print('The shape of test data:' , df_test.shape)
df_crude_oil.tail()
df_train.tail()
df_test.tail()
df_crude_oil.describe()
plt.figure(figsize=(20,10))

plt.plot(df_crude_oil, color = 'red')

plt.xlabel('Date in year')

plt.ylabel('Oill Price ')

plt.show()

columns = [844]

df = df_train[df_train.columns[columns]]

plt.figure(figsize=(20,10))

plt.plot(df, color='blue')

plt.xlabel('Date')

plt.ylabel('price')
cols = [843]

df_1 = df_train[df_train.columns[cols]]

plt.figure(figsize=(20,10))

plt.plot(df_1, color='red')

plt.xlabel('Date')

plt.ylabel('world_new_deaths')
cols = [842]

df_2 = df_train[df_train.columns[cols]]

plt.figure(figsize=(20,10))

plt.plot(df_2, color='red')

plt.xlabel('Date')

plt.ylabel('World_total_deaths')
cols = [841]

df_3 = df_train[df_train.columns[cols]]

plt.figure(figsize=(20,10))

plt.plot(df_3, color='red')

plt.xlabel('Date')

plt.ylabel('World_new_cases')
cols = [840]

df_3 = df_train[df_train.columns[cols]]

plt.figure(figsize=(20,10))

plt.plot(df_3, color='red')

plt.xlabel('Date')

plt.ylabel('World_total_cases')
cols = [840,841,842,843,844]

df_corr= df_train[df_train.columns[cols]]



corr = df_corr.corr()



fig = plt.figure(figsize=(20,10))

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(df_corr.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(df_corr.columns)

ax.set_yticklabels(df_corr.columns)

plt.show()
cols = [836,837,838,839, 844]

df_corr= df_train[df_train.columns[cols]]



corr = df_corr.corr()

fig =  plt.figure(figsize=(20,10))

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(df_corr.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(df_corr.columns)

ax.set_yticklabels(df_corr.columns)

plt.show()
cols = [404,405,406,407, 844]

df_corr= df_train[df_train.columns[cols]]



corr = df_corr.corr()

fig =  plt.figure(figsize=(20,10))

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(df_corr.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(df_corr.columns)

ax.set_yticklabels(df_corr.columns)

plt.show()