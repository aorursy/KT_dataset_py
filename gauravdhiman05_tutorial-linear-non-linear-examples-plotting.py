%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
def log_func(x):
    if x > 0:
        return math.log(x,10)
    else:
        return np.nan
df = pd.DataFrame(index=range(-200,200))
df.shape
title = ['Linear 3*X','Quadratic X^2',
         'Cubic X^3','Absolute abs(X)',
         'sine(X)','log(X)',
         'Exponential 2^X']

df['linear']=df.index.map(lambda x: 3*x)
df['quadratic'] = df.index.map(lambda x: x**2)
df['cubic'] = df.index.map(lambda x: x**3)
df['abs'] = df.index.map(lambda x: abs(x))
df['sine'] = np.sin(np.arange(-20,20,.1))
df['log'] = df.index.map(log_func)
df['exponential'] = df.index.map(lambda x: 2.0**x)
df.head()
df.tail()
fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=False)
axx = axs.ravel()
for i in range(df.shape[1]-3):
    axx[i].set_title(title[i])
    df[df.columns[i]].plot(ax=axx[i])
    axx[i].set_xlabel('X')
    if i % 2 == 1 :
        axx[i].yaxis.tick_right()
    axx[i].grid()
plt.figure(figsize=(10,3))
plt.plot(df['exponential'])
plt.title('Exponential 2^X')
plt.xlim(-10,20)
plt.ylim(0,100000)
plt.xlabel('X')
plt.grid()
plt.figure(figsize=(10,3))
plt.plot(df['log'])
plt.title('Log(X)')
plt.xlim(-10,200)
plt.xlabel('X')
plt.grid()
plt.figure(figsize=(10,3))
plt.plot(np.arange(-200,200), df['sine'])
plt.title('Sine(X)')
plt.xlabel('X')
plt.grid()
