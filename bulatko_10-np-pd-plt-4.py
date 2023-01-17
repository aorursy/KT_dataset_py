import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
df = pd.read_csv('../input/sputnik/train.csv')

df['Datetime'] = pd.to_datetime(df.epoch,format='%Y-%m-%d %H:%M:%S')

df.index = df.Datetime

df = df.drop(['epoch', 'Datetime'], axis=1)



df.head()
import random

i = random.randint(0, 599)



y1 = df[df['sat_id'] == i]['y'].iloc[:24].values

y2 = (df[df['sat_id'] == i]['y'].iloc[24:48]).values

plt.plot(np.linspace(0, 1, 24), y1)

plt.plot(np.linspace(0, 1, 24), y2)
import random

random.seed(249)

s_id = random.randint(0, 599)

coord = ['x', 'y', 'z'][random.randint(0, 2)]



df[df.sat_id == s_id][coord].plot()

print(s_id, coord)
for sat_id in np.unique(df['sat_id'].values):

    print(sat_id, end = ' ')

    frame = df[df['sat_id'] == sat_id]

    for v in ['x', 'y', 'z']:

        e = frame[v].values

        t = frame['type'].values

        for i in range(len(frame[v])):

            if t[i] == 'test':

                e[i] = e[i - 24] + (e[i - 24] - e[i - 48])

    df[df['sat_id'] == sat_id] = frame
s_id = random.randint(0, 599)

coord = ['x', 'y', 'z'][random.randint(0, 2)]



df[df.sat_id == s_id][coord].plot()

print(s_id, coord)
df['error']  = np.linalg.norm(df[['x', 'y', 'z']].values - df[['x_sim', 'y_sim', 'z_sim']].values, axis=1)
ans = df[df['type'] == 'test'][['id', 'error']]

ans



ans.to_csv('ans.csv', index=False)