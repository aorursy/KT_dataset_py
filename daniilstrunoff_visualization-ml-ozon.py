import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

%matplotlib inline
raw_data = pd.read_csv('../input/london-bike-sharing-dataset/london_merged.csv')
plt.figure(figsize=(14, 10))

a = plt.hist(raw_data['t1'][raw_data['t1'] % 1 == 0],bins=30, color = '#fdbb84')

a = plt.hist(raw_data['t1'][raw_data['t1'] % 1 == 0.5],bins=30, color = '#43a2ca')

plt.xlabel('температура', fontsize=20)
def get_color(data, n=10000, r=20):

    color = np.array([0]*n)

    for i, x in zip(range(n), data[:n]) :

        print(i, end='\r')

        color[np.linalg.norm(data[:n] - x, axis=1) < r] += 1

    return color
color = get_color(raw_data[['wind_speed','hum']].values, r=10)
n=10000

plt.figure(figsize=(16, 10))

plt.scatter(raw_data['wind_speed'][:n] + np.random.normal(0, 0.3, n),

            raw_data['hum'][:n]+ np.random.normal(0, 0.3, n),

            c = color,alpha=0.6,s=color/100, cmap='viridis')



plt.colorbar()

plt.xlabel('wind speed', fontsize=20)

plt.ylabel('humidity', fontsize=20)

plt.grid()

plt.show()
raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'])
from scipy.sparse.linalg import svds

cr = raw_data.corr()

ii, _, _ = svds(cr, k=1)

ii = np.argsort(ii[:,0])

plt.imshow(cr.iloc[ii, ii], interpolation='none')

h = plt.colorbar()

jj = np.arange(len(ii))[np.argsort(ii)]

x = plt.xticks(jj, raw_data.columns[1:], rotation=70)

y = plt.yticks(jj, raw_data.columns[1:], rotation=0)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

X = pca.fit_transform(raw_data[raw_data.columns[2:]])
plt.figure(figsize=(14,10))

plt.scatter(X[:,0][(raw_data['cnt'] < 4000) & (raw_data['cnt'] > 200)],

            X[:, 1][(raw_data['cnt'] < 4000) & (raw_data['cnt'] > 200)],

            c=raw_data['cnt'][(raw_data['cnt'] < 4000) & (raw_data['cnt'] > 200)]/100,

            alpha=0.6)

plt.colorbar()
raw_data['day_of_week'] = raw_data['timestamp'].dt.dayofweek
raw_data['month'] = raw_data['timestamp'].dt.month
raw_data['hour'] = raw_data['timestamp'].dt.hour
df = pd.pivot_table(raw_data[["day_of_week", "hour", "cnt"]], 

                     index="day_of_week", 

                     columns="hour", 

                     values="cnt")
import calendar
plt.figure(figsize=(29,10))

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

h = sns.heatmap(data=df, cmap='viridis', annot=True, linewidths=.5)

x = plt.xticks(np.arange(24)+0.5, np.arange(24))

y = plt.yticks(np.arange(7)+0.5, days, rotation=0)

plt.xlabel('hour', fontsize=20)

plt.ylabel('day of week', fontsize=20)
plt.figure(figsize=(18,10))

index_list = ((raw_data['hour']==8) &

              (raw_data['day_of_week'] != 5) & 

              (raw_data['day_of_week'] != 6) & 

              (raw_data['timestamp'].dt.year == 2015))



plt.plot(raw_data['timestamp'][index_list], raw_data['cnt'][index_list])



plt.fill_between(raw_data['timestamp'][index_list], 0, 7000, where=raw_data['cnt'][index_list] < 1000,

                facecolor='red', alpha=0.5, interpolate=True)

plt.fill_between(raw_data['timestamp'][index_list], 0, 7000, where=raw_data['cnt'][index_list] > 4800,

                facecolor='green', alpha=0.5, interpolate=True)

#plt.xticks(calendar.month_name[1:13], rotation=20)

plt.grid()
raw_data[index_list][raw_data['cnt'][index_list] > 5000]