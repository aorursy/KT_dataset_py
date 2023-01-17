# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/2017.csv')
data.info()
#correlation map

data.corr()
#corr map

f,ax =plt.subplots(figsize=(15,15))

sns.heatmap(data.corr(),annot=True, linewidth=.5, fmt='.1f', ax=ax)

plt.show()
data.head(10)
data.columns
#Line Plot x=Whisker.high y=Happiness.Score  Whisker.high&Happiness.Score corr=0.99

data['Whisker.high'].plot(kind='line',color='g',label='Whisker.high', linewidth=3, alpha=0.5, grid=True, linestyle=':')

data['Happiness.Score'].plot(kind='line', color='r', label='Happiness.Score', linewidth=3, alpha=0.5, grid=True, linestyle='-.')

plt.legend(loc='lower right')

plt.xlabel('Whisker.high')

plt.ylabel('Happiness.Score')

plt.title('Whisker.high and Happiness.Score Line Plot Review')

plt.show()
#Line Plot  Whisker.high - Whisker.low - Happiness.Score Relations & Economy..GDP.per.Capita. - Health..Life.Expectancy. - Family Relations

data['Whisker.high'].plot(kind='line',color='g',label='Whisker.high', linewidth=3, alpha=0.8, grid=True, linestyle=':', figsize=(12,8))

data['Whisker.low'].plot(kind='line', color='b', label='Whisker.low', linewidth=3, alpha=0.8, grid=True, linestyle='-')

data['Happiness.Score'].plot(kind='line', color='r', label='Happiness.Score', linewidth=3, alpha=0.8, grid=True, linestyle='-.')

data['Economy..GDP.per.Capita.'].plot(kind='line', color='yellow', label='Economy..GDP.per.Capita.', linewidth=3, alpha=0.8, grid=True, linestyle='--')

data['Health..Life.Expectancy.'].plot(kind='line', color='purple', label='Health..Life.Expectancy.', linewidth=3, alpha=0.8, grid=True, linestyle=':')

data['Family'].plot(kind='line', color='orange', label='Family', linewidth=3, alpha=0.8, grid=True, linestyle='-.')

plt.legend(loc='lower right')

plt.title('Whisker.high - Whisker.low - Happiness.Score Relations & Economy..GDP.per.Capita. - Health..Life.Expectancy. - Family Relations')

plt.show()
data['Happiness.Score'].plot(kind='hist',bins=50, figsize=(12,12))

plt.show()
# Scatter Plot #corr=0.99

# x = Happiness.Score, y = Whisker.low

data.plot(kind='scatter', x='Happiness.Score', y='Whisker.low', alpha=0.5, color='b', figsize=(10,5))

plt.xlabel('Happiness.Score')

plt.ylabel('Whisker.low')

plt.title('Happiness.Score and Whisker.low Scatter Plot Review')

plt.show()