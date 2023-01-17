%matplotlib inline

import feather

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from numba import jit

from tqdm import tqdm_notebook as tqdm

import warnings

warnings.filterwarnings('ignore')
train_df = feather.read_dataframe('../input/ashrae-feather/train.ft')
# loss functions



# Root Mean Squared Error

def rmse(ytrue, ypred):

    return np.sqrt(np.mean(np.square(ypred - ytrue), axis=0))



# Root Mean Squared Log Error

def rmsle(ytrue, ypred):

    return np.sqrt(np.mean(np.square(np.log1p(ypred) - np.log1p(ytrue)), axis=0))



# Function to find a singular minimum value

def minimize(i):

    return rmsle(train_df['meter_reading'].values, np.full(len(train_df), i, dtype=np.float32))
print(f"Mean Meter Reading: {np.mean(train_df['meter_reading'].values):.2f}")
%%time

m = []

for i in range(2117):

    m.append(minimize(i))
plt.title('Finding the minimum $x$')

plt.ylabel('RMSLE')

plt.xlabel('$x$')

plt.xscale('log')

plt.plot(np.arange(2117), m)

plt.show()
print(f'Min RMSLE of {np.min(m):.2f} is found at x={np.argmin(m)}')
def min_x(i, y_true):

    return rmsle(y_true, np.full(len(y_true), i, dtype=np.float32))
train_df.groupby('meter')['meter_reading'].agg(np.mean)
%%time

m0, m1, m2, m3 = [],[],[],[]

for i in tqdm(range(0,1000,10)):

    m0.append(min_x(i, train_df[train_df['meter'] == 0]['meter_reading'].values))

    

for i in tqdm(range(0,2000,10)):

    m1.append(min_x(i, train_df[train_df['meter'] == 1]['meter_reading'].values))

    

for i in tqdm(range(0,20000,10)):

    m2.append(min_x(i, train_df[train_df['meter'] == 2]['meter_reading'].values))

    

for i in tqdm(range(0,2000,10)):

    m3.append(min_x(i, train_df[train_df['meter'] == 3]['meter_reading'].values))
plt.figure(figsize=(12,12))



plt.subplot(221)

plt.title('Finding the minimum $x_0$')

plt.ylabel('RMSLE')

plt.xlabel('$x_0$')

plt.xscale('log')

plt.plot(np.arange(0,1000,10), m0)



plt.subplot(222)

plt.title('Finding the minimum $x_1$')

plt.ylabel('RMSLE')

plt.xlabel('$x_1$')

plt.xscale('log')

plt.plot(np.arange(0,2000,10), m1)



plt.subplot(223)

plt.title('Finding the minimum $x_2$')

plt.ylabel('RMSLE')

plt.xlabel('$x_2$')

plt.xscale('log')

plt.plot(np.arange(0,20000,10), m2)



plt.subplot(224)

plt.title('Finding the minimum $x_3$')

plt.ylabel('RMSLE')

plt.xlabel('$x_3$')

plt.xscale('log')

plt.plot(np.arange(0,2000,10), m3)



plt.show()
print(f'x0: Min RMSLE of {np.min(m0):.2f} is found at x={np.argmin(m0)*10}')

print(f'x1: Min RMSLE of {np.min(m1):.2f} is found at x={np.argmin(m1)*10}')

print(f'x2: Min RMSLE of {np.min(m2):.2f} is found at x={np.argmin(m2)*10}')

print(f'x3: Min RMSLE of {np.min(m3):.2f} is found at x={np.argmin(m3)*10}')
%%time

# do this one more time to get to a closer minimum for each x

m0, m1, m2, m3 = [],[],[],[]

for i in tqdm(range(40,60)):

    m0.append(min_x(i, train_df[train_df['meter'] == 0]['meter_reading'].values))

    

for i in tqdm(range(60,80)):

    m1.append(min_x(i, train_df[train_df['meter'] == 1]['meter_reading'].values))

    

for i in tqdm(range(160,180)):

    m2.append(min_x(i, train_df[train_df['meter'] == 2]['meter_reading'].values))

    

for i in tqdm(range(10,30)):

    m3.append(min_x(i, train_df[train_df['meter'] == 3]['meter_reading'].values))
print(f'x0: Min RMSLE of {np.min(m0):.3f} is found at x0={np.argmin(m0)+40}')

print(f'x1: Min RMSLE of {np.min(m1):.3f} is found at x1={np.argmin(m1)+60}')

print(f'x2: Min RMSLE of {np.min(m2):.3f} is found at x2={np.argmin(m2)+160}')

print(f'x3: Min RMSLE of {np.min(m3):.3f} is found at x3={np.argmin(m3)+10}')
p0 = np.full(len(train_df[train_df['meter']==0]), 52)

p1 = np.full(len(train_df[train_df['meter']==1]), 69)

p2 = np.full(len(train_df[train_df['meter']==2]), 167)

p3 = np.full(len(train_df[train_df['meter']==3]), 27)

g0 = train_df[train_df['meter']==0]['meter_reading'].values

g1 = train_df[train_df['meter']==1]['meter_reading'].values

g2 = train_df[train_df['meter']==2]['meter_reading'].values

g3 = train_df[train_df['meter']==3]['meter_reading'].values

p = []

g = []

p += list(p0)

p += list(p1)

p += list(p2)

p += list(p3)

g += list(g0)

g += list(g1)

g += list(g2)

g += list(g3)
print(f"Total RMSLE: {rmsle(np.array(g), np.array(p)):.3f}")
%%time

sub = pd.read_csv('../input/ashrae-energy-prediction/sample_submission.csv', dtype={'row_id': 'int32', 'meter_reading': 'int8'})

sub['meter_reading'] = 62

sub['meter_reading'] = sub['meter_reading'].astype('int8')
%%time

sub.to_csv('submission.csv', index=False)