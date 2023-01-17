import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

data=pd.read_csv('../input/temperatures-of-india/temperatures.csv')

data.head()
data.describe()
fig, axs = plt.subplots(ncols=2,figsize=(20,10))

sns.scatterplot(x=data['YEAR'], y=data['ANNUAL'], data=data, ax=axs[0])

sns.scatterplot(x=data['YEAR'], y=data['JAN-FEB'],data=data, ax=axs[1])

fig, axs = plt.subplots(ncols=2,figsize=(20,10))

sns.scatterplot(x=data['YEAR'],y=data['MAR-MAY'],data=data, ax=axs[0])

sns.scatterplot(x=data['YEAR'],y=data['JUN-SEP'],data=data, ax=axs[1])

fig, axs = plt.subplots(ncols=5,figsize=(20,10))

sns.boxplot(data['ANNUAL'],orient="v", data=data, ax=axs[0])

sns.boxplot(data['JAN-FEB'],orient="v", data=data, ax=axs[1])

sns.boxplot(data['MAR-MAY'],orient="v", data=data, ax=axs[2])

sns.boxplot(data['JUN-SEP'],orient="v", data=data, ax=axs[3])

sns.boxplot(data['OCT-DEC'],orient="v", data=data, ax=axs[4])