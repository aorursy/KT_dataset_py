import pandas as pd

import matplotlib.pyplot as plt

import plotly.plotly as py

import numpy as np

import seaborn as sns

import calendar

%matplotlib inline

df = pd.read_csv('../input/911.csv')

df.head()
reason = np.unique(df['title'])
reason.size
DATA = np.zeros((df.shape[0],6),dtype='O')

DATA[:,0] = df['lng'].values

DATA[:,1] = df['lat'].values

DATA[:,4] = df['title'].values

DATA[:,5] = df['twp'].values

for i in range(DATA.shape[0]):

    DATA[i,2] = df['timeStamp'].values[i][:10]

    DATA[i,3] = df['timeStamp'].values[i][10:]

    sp = DATA[i,3].split(':')

    DATA[i,3] = (int(sp[0])*3600 + int(sp[1])*60 + int(sp[2]))/3600
new_data = np.zeros(reason.size,dtype = 'O')

for i in range(reason.size):

    new_data[i] = DATA[np.where(DATA[:,4] == reason[i])]
week = np.array(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
for i in range(new_data.shape[0]):

    for j in range(new_data[i].shape[0]):

        w = np.array(new_data[i][j,2].split('-')).astype(int)

        new_data[i][j,0] = week[calendar.weekday(w[0],w[1],w[2])]
for i in range(reason.size):

    if new_data[i][:,3].size > 1700:

        sns.plt.figure(figsize=(12,4))

        sns.plt.title(new_data[i][0][-2])

        sns.plt.xlabel("Week day")

        sns.plt.ylabel(new_data[i][0][-2])

        print("Number of calls with " + new_data[i][0][-2] + " "+ str(new_data[i][:,3].size))

        sns.countplot((new_data[i][:,0]),order = week)