%matplotlib inline



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



data = pd.read_csv('../input/uber-raw-data-janjune-15.csv')



print(data.shape[0])
per_date = data.groupby(['Pickup_date'], as_index=False).agg(['count'])

per_date.reset_index(level=['Pickup_date'], inplace=True)

per_date = per_date.ix[:,0:2]

per_date.columns = ['date', 'count']

print(per_date.shape[0])

print (per_date.head())
per_date['date'] = per_date['date'].apply(lambda x: pd.Timestamp(x))

per_date['dayhour'] = per_date['date'].dt.strftime('%y.%m.%d %H')

per_date = per_date[['dayhour', 'count']].groupby(['dayhour'], as_index=False)['count'].agg(['sum'])

per_date.reset_index(level=['dayhour'], inplace=True)

per_date.columns = ['dayhour', 'total']



print(per_date.head())
per_date['dayhour'] = per_date['dayhour'].apply(lambda x: pd.Timestamp(x))

per_date['hour'] = per_date['dayhour'].dt.strftime('%H')

per_date['weekday'] = per_date['dayhour'].dt.strftime('%a')

per_date = per_date.groupby(['weekday', 'hour'], as_index=False)['total'].agg(['mean'])

per_date.reset_index(level=['weekday', 'hour'], inplace=True)



print(per_date.head())
sequence = {'Mon': 6, 'Tue': 5, 'Wed': 4, 'Thu': 3, 'Fri': 2, 'Sat': 1, 'Sun': 0}

per_date['daynum'] = per_date['weekday'].apply(lambda x: sequence[x])

per_date = per_date.sort_values(by=['daynum', 'hour'])

per_date = per_date[['weekday', 'hour', 'mean']]



print(per_date.head())
mat=per_date['mean'].values.reshape(7,24)

plt.title('Mean Number of Journeys by Weekday and Hour')

plt.xlim([0, 24])

plt.ylim([0, 7])

plt.xlabel('Hour')

plt.ylabel('Week Day')

plt.yticks(np.arange(0.5, 7.5, 1), per_date['weekday'].unique())

plt.pcolor(mat,cmap=plt.cm.Reds)

plt.colorbar()

plt.show()