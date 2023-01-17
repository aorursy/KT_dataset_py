# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/2017.csv")
data.columns
data.info()
data.corr()
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f')

plt.show()
data.head(10)

data.tail(10)
data.columns
plt.plot(data["Happiness.Rank"],data["Trust..Government.Corruption."],color="red",label="trust")

plt.plot(data["Happiness.Rank"],data["Freedom"],color="blue",label="Freedom")

plt.legend()

plt.xlabel("x")

plt.ylabel("y")

plt.show()
data.plot(kind='scatter', x='Trust..Government.Corruption.', y='Happiness.Score',alpha = 0.7,color = 'red')

plt.title('trust-hapines scater plot') 

plt.show()
data["Happiness.Score"].plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
data[data["Happiness.Score"]>7]
data[(data['Happiness.Score']>7) & (data['Whisker.low']<7)]

data[np.logical_and(data['Happiness.Score']>7, data['Economy..GDP.per.Capita.']> 1.47 )]