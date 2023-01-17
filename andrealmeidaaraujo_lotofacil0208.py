# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
sns.set()
import re
import datetime
def read(data_file, sep=','):
    try:
        df = pd.read_csv(data_file, sep)
        return df
    except Exception as e:
        print(e)
        
df = read('/kaggle/input/mega.csv')
df.head(5)
plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
df.describe()
_x = [i for i in range(0,2426)]

def plot_charts():
    for column in df.columns.values:
        plot_chart(column)
    
def plot_chart(column):
    _y = df[column]
    plt.figure()
    plt.scatter(_x, _y, marker='.')
    plt.tight_layout()
    plt.title(column)
    plt.show()
    
plot_charts()
def get_tuples():
    return list(zip(df['n1'], df['n2'], df['n3'], df['n4'], df['n5'], df['n6'],df['n7'],df['n8'],df['n9'],df['n10'],df['n11'],df['n12'],df['n13'],df['n14'],df['n15']))

_min = df.min(axis=1)
_max = df.max(axis=1)
_mean = df.mean(axis=1)
_std = df.std(axis=1)
_var = df.var(axis=1)
_median = df.median(axis=1)
Q1 = df.quantile(0.25, axis=1)
Q3 = df.quantile(0.75, axis=1)

#magnitudes = np.ndarray((2425,)) 
#i = 0
#for t in get_tuples():
#    basic = np.array(t)
#    magnitudes[i] = np.linalg.norm(np.square(basic))
#    i = i + 1
    
#df['abs'] = pd.Series(magnitudes)
df['min'] = _min
df['max'] = _max
df['mean'] = _mean
df['std'] = _std
df['var'] = _var
df['median'] = _median
df['iqr'] = Q3 - Q1

df.head(5)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model

N_TIME_STEPS = 15
N_FEATURES = 15
step = 15
segments = []
labels = []
for i in range(0, len(df) - N_TIME_STEPS, step):
    n1s = df['n1'].values[i: i + N_TIME_STEPS]
    n2s = df['n2'].values[i: i + N_TIME_STEPS]
    n3s = df['n3'].values[i: i + N_TIME_STEPS]
    n4s = df['n4'].values[i: i + N_TIME_STEPS]
    n5s = df['n5'].values[i: i + N_TIME_STEPS]
    n6s = df['n6'].values[i: i + N_TIME_STEPS]
    n7s = df['n7'].values[i: i + N_TIME_STEPS]
    n8s = df['n8'].values[i: i + N_TIME_STEPS]
    n9s = df['n9'].values[i: i + N_TIME_STEPS]
    n10s = df['n10'].values[i: i + N_TIME_STEPS]
    n11s = df['n11'].values[i: i + N_TIME_STEPS]
    n12s = df['n12'].values[i: i + N_TIME_STEPS]
    n13s = df['n13'].values[i: i + N_TIME_STEPS]
    n14s = df['n14'].values[i: i + N_TIME_STEPS]
    n15s = df['n15'].values[i: i + N_TIME_STEPS]
    
    #mins = df['min'].values[i: i + N_TIME_STEPS]
    #maxs = df['max'].values[i: i + N_TIME_STEPS]
    #means = df['mean'].values[i: i + N_TIME_STEPS]
    #stds = df['std'].values[i: i + N_TIME_STEPS]
    #var = df['var'].values[i: i + N_TIME_STEPS]
    #medians = df['median'].values[i: i + N_TIME_STEPS]
    #iqrs = df['iqr'].values[i: i + N_TIME_STEPS]    
    
    if i + N_TIME_STEPS + 1 < df.shape[0]:
        label = df.iloc[i + N_TIME_STEPS + 1]

    #segments.append([n1s, n2s, n3s, n4s, n5s, n6s, 
     #                mins, maxs, means, stds, var, medians, iqrs])
    segments.append([n1s, n2s, n3s, n4s, n5s, n6s, n7s, n8s, n9s, n10s, n11s, n12s, n13s, n14s, n15s])

    labels.append(label)

print(np.array(segments).shape)
reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
labels = np.asarray(labels)
model = Sequential()
model.add(LSTM(10, input_shape=(15,15)))
model.add(Dense(13, activation='linear'))

model.compile(loss='mse', optimizer='adam')

#X,y = get_train()
model.fit(reshaped_segments, labels, epochs=300, shuffle=False, verbose=0)
yhat = model.predict(reshaped_segments, verbose=0)
print(yhat[0])