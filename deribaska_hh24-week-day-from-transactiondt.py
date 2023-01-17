import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure
df = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")
print('Train dataset is '+str(round((df.TransactionDT.max() - df.TransactionDT.min())/60/60/24/7))+' week period')
pd.value_counts(df['isFraud'], sort = True)
fraud = df[df.isFraud == 1].TransactionDT

normal = df[df.isFraud == 0].TransactionDT.sample(n = len(fraud)) # the same size like fraud dataset (20663)
df_dow = pd.DataFrame(columns = ['TimeDif','SumDif'])

for i in range(24):

    nwd = pd.to_datetime(normal + i * 3600, unit='s').dt.day_name()

    fwd = pd.to_datetime(fraud + i * 3600, unit='s').dt.day_name()

    df_dow = df_dow.append({'TimeDif':i,'SumDif':abs(1 - nwd.groupby(nwd).count()/fwd.groupby(fwd).count()).sum()},ignore_index=True)
figure(figsize=(16,5))

plt.plot(df_dow['SumDif'])

plt.title('Fraud/Not-Fraud with intraday shifting')

plt.xlabel('Shifted time (Hours)')

plt.ylabel('Sum of abs(1-normal/fraud) per day');

plt.show()
best_dif = df_dow[df_dow.SumDif == df_dow.SumDif.max()].TimeDif.values * 3600

worst_dif = df_dow[df_dow.SumDif == df_dow.SumDif.min()].TimeDif.values * 3600

print('Optimal shifting is ' + str(int(best_dif[0]/60/60)) + ' hours')

print('Worst shifting is ' + str(int(worst_dif[0]/60/60)) + ' hours')
nwd = pd.to_datetime(normal + best_dif, unit='s').dt.dayofweek

fwd = pd.to_datetime(fraud + best_dif, unit='s').dt.dayofweek

fig,ax1 = plt.subplots()

ax1.plot(nwd.groupby(nwd).count(),label='Normal')

ax1.plot(fwd.groupby(fwd).count(),label='Fraud')

plt.legend(loc='upper right')

plt.title('Day of week and transactions per day (optimal shifted in time )')

plt.xlabel('Day of Week')

plt.ylabel('Transactions Per Day')

ax1.set_facecolor('seashell')

fig.set_figwidth(16)

fig.set_figheight(2)

fig.set_facecolor('floralwhite')

plt.show()

nwd = pd.to_datetime(normal, unit='s').dt.dayofweek

fwd = pd.to_datetime(fraud, unit='s').dt.dayofweek

fig,ax1 = plt.subplots()

ax1.plot(nwd.groupby(nwd).count(),label='Normal')

ax1.plot(fwd.groupby(fwd).count(),label='Fraud')

plt.legend(loc='upper right')

plt.title('Day of week and transactions per day (not shifted in time )')

plt.xlabel('Day of Week')

plt.ylabel('Transactions Per Day')

ax1.set_facecolor('seashell')

fig.set_figwidth(16)

fig.set_figheight(2)

fig.set_facecolor('floralwhite')

plt.show()

nwd = pd.to_datetime(normal + worst_dif, unit='s').dt.dayofweek

fwd = pd.to_datetime(fraud + worst_dif, unit='s').dt.dayofweek

fig,ax1 = plt.subplots()

ax1.plot(nwd.groupby(nwd).count(),label='Normal')

ax1.plot(fwd.groupby(fwd).count(),label='Fraud')

plt.legend(loc='upper right')

plt.title('Day of week and transactions per day (wrong shifted in time )')

plt.xlabel('Day of Week')

plt.ylabel('Transactions Per Day')

ax1.set_facecolor('seashell')

fig.set_figwidth(16)

fig.set_figheight(2)

fig.set_facecolor('floralwhite')

plt.show()
x1 = np.arange(1, 8) - .3

nwd = pd.to_datetime(normal + best_dif, unit='s').dt.dayofweek

fwd = pd.to_datetime(fraud + best_dif, unit='s').dt.dayofweek

y1 = abs(1 - nwd.groupby(nwd).count()/fwd.groupby(fwd).count()).values

x2 = x1 + .3

nwd = pd.to_datetime(normal, unit='s').dt.dayofweek

fwd = pd.to_datetime(fraud, unit='s').dt.dayofweek

y2 = abs(1 - nwd.groupby(nwd).count()/fwd.groupby(fwd).count()).values

x3 = x2 + .3

nwd = pd.to_datetime(normal + worst_dif, unit='s').dt.dayofweek

fwd = pd.to_datetime(fraud + worst_dif, unit='s').dt.dayofweek

y3 = abs(1 - nwd.groupby(nwd).count()/fwd.groupby(fwd).count()).values

fig,ax = plt.subplots()

ax.bar(x1, y1, width = 0.3, label='Optimal shifted')

ax.bar(x2, y2, width = 0.3, label='Not shifted')

ax.bar(x3, y3, width = 0.3, label='Not optimal shifted')

plt.legend(loc='upper right')

plt.title('Day of week and transactions per day (shifted in time )')

plt.xlabel('Day of Week')

plt.ylabel('Sum of abs(1 - normal/fraud) transactions per day')



ax.set_facecolor('seashell')

fig.set_figwidth(16)

fig.set_figheight(6)

fig.set_facecolor('floralwhite')



plt.show()
figure(figsize=(18,5))

plt.hist(pd.to_datetime(fraud, unit='s').dt.hour,np.linspace(0,24,25),alpha=.5,density = True,label='Normal')

plt.hist(pd.to_datetime(normal, unit='s').dt.hour,np.linspace(0,24,25),alpha=.5,density = True,label='Fraud')

plt.legend(loc = 'lower right')

plt.title('Transactions per hour')

plt.xlabel('Hour of Day')

plt.ylabel('Transactions')

plt.show()
df_hh24 = pd.DataFrame(columns = ['TimeDif','SumDif'])

for i in range(60):

    nhh24 = pd.to_datetime(normal + i * 60, unit='s').dt.hour

    fhh24 = pd.to_datetime(fraud + i * 60, unit='s').dt.hour

    df_hh24 = df_hh24.append({'TimeDif':i,'SumDif':abs(1 - nhh24.groupby(nhh24).count()/fhh24.groupby(fhh24).count()).sum()},ignore_index=True)

figure(figsize=(18,5))

plt.plot(df_hh24['SumDif'])

plt.title('Transactions per minute')

plt.xlabel('Minute of Hour')

plt.ylabel('Sum of abs(1-normal/fraud) transactions')

plt.show()
# Perhaps there are reasonable to add two additional features in the model:

# pd.to_datetime(df.TransactionDT + 7 * 60 * 60, unit='s').dt.day_name() (shifted forward on 7 hours);

# pd.to_datetime(normal + i * 3600, unit='s').dt.hour