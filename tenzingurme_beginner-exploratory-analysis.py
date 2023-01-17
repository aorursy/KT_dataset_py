import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # interactive plotting

import matplotlib.pyplot as plt # Basic plotting

from datetime import datetime

import os

reservoir = pd.read_csv('../input/chennai_reservoir_levels.csv')

rain = pd.read_csv('../input/chennai_reservoir_rainfall.csv')
reservoir.head()
rain.head()
reservoir.info()
rain.info()
reservoir.describe()
rain.describe()
sns.set_style('dark')

reservoir.POONDI.plot(figsize=(20,10))

plt.ylabel('Rain Level in Poondi')

plt.show()
reservoir.CHOLAVARAM.plot(figsize=(20,10), c='g')

plt.ylabel('Rain Level in CHOLAVARAM')

plt.show()
reservoir.REDHILLS.plot(figsize=(20,10),c='r')

plt.ylabel('Rain Level in REDHILLS')

plt.show()
reservoir.CHEMBARAMBAKKAM.plot(figsize=(20,10),c='k')

plt.ylabel('Rain Level in CHEMBARAMBAKKAM')

plt.show()
fig, (ax1, ax2) = plt.subplots( 1, 2, figsize=(15,5))

ax1.plot(rain.REDHILLS, c='r')

ax2.plot(rain.CHEMBARAMBAKKAM, c='k')

plt.show()
fig, (ax1, ax2) = plt.subplots( 1, 2, figsize=(15,5))

ax1.plot(rain.REDHILLS, c='r')

ax2.plot(rain.POONDI, c='b')

plt.show()
fig, (ax1, ax2) = plt.subplots( 1, 2, figsize=(15,5))

ax1.plot(rain.CHOLAVARAM, c='g')

ax2.plot(rain.CHEMBARAMBAKKAM, c='k')

plt.show()
fig, (ax1, ax2) = plt.subplots( 1, 2, figsize=(15,5))

ax1.plot(rain.CHOLAVARAM, c='g')

ax2.plot(rain.POONDI, c='b')

plt.show()
fig, (ax1, ax2) = plt.subplots( 1, 2, figsize=(15,5))

ax1.plot(rain.REDHILLS, c='r')

ax2.plot(rain.POONDI, c='b')

plt.show()
corr = reservoir.corr()

corr.style.background_gradient(cmap='coolwarm')
fig, ax = plt.subplots(figsize=(8,8))

sns.heatmap(reservoir.corr(), annot=True, linewidth=.5, fmt= '.1f',ax=ax)

plt.show()
rain.plot(figsize=(20,10), linewidth=3, fontsize=15)

plt.xlabel('Year', fontsize=15)

plt.ylabel('Rain Level', fontsize=15);
reservoir.POONDI.plot(kind='line', c='g', label='POONDI', linewidth=1, alpha=0.5, grid=True, linestyle=':')

reservoir.REDHILLS.plot(kind='line', c='r', label='REDHILLS', linewidth=1, alpha=0.5, grid=True, linestyle=':')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('REDHILLS AND POONDI PLOT')

plt.show()