# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
tips= pd.read_csv('../input/tips.csv')

# tips= sns.load_dataset('../input/tips.csv')
tips.head(5)
sns.barplot(x='day', y='tip', data=tips)
sns.barplot(x='day', y='total_bill', data=tips)

sns.barplot(x='day', y='tip', data=tips, hue='sex', palette='husl')
sns.barplot(x='day', y='total_bill', data=tips, hue='smoker', palette= 'muted')
sns.barplot(x='total_bill', y='day', data=tips, palette='winter_r')

sns.barplot(x='day', y='tip', data=tips, palette='dark', order=['Sat','Fri','Sun','Thur'])
from numpy import median
sns.barplot(x='day', y='total_bill', data=tips, palette='colorblind', estimator=median)
sns.barplot(x='smoker', y='tip', data=tips, estimator=median, hue='sex', palette='coolwarm')
sns.barplot(x='smoker', y='tip', data=tips, ci=99)
sns.barplot(x='smoker', y='tip', data=tips, ci=68)
sns.barplot(x='smoker', y='tip', data=tips, ci=34, palette='winter_r', estimator=median)
sns.barplot(x='day', y='total_bill', data=tips, palette='dark', capsize=0.3)
sns.barplot(x='day', y='total_bill', data=tips, palette='husl', capsize=0.9)
sns.barplot(x='day', y='total_bill', data=tips, palette='winter_r', capsize=0.1)
sns.barplot(x='day', y='total_bill', data=tips, hue='sex', palette='dark', capsize=0.1)
tips.head(5)
sns.boxplot(x=tips['size'])
sns.boxplot(x=tips['total_bill'])
sns.boxplot(x='sex', y='total_bill', data=tips, palette= 'dark')
sns.boxplot(x='day', y='total_bill', data=tips)
sns.boxplot(x='day', y='total_bill', data=tips, hue='sex', palette='spring')
sns.boxplot(x='day', y='total_bill', data=tips, hue='smoker', palette='coolwarm')
sns.boxplot(x='day', y='total_bill', data=tips, hue='time', palette= 'spring')
iris= pd.read_csv('../input/iris.csv')
tips.head(5)
sns.swarmplot(x='day', y='total_bill', data=tips)
sns.swarmplot(x='day', y='tip', data=tips)
tips.head(5)
sns.boxplot(x='day', y='total_bill', data=tips, palette='husl')

sns.swarmplot(x='day', y='total_bill', data=tips)
sns.boxplot(x='day', y='total_bill', data=tips, palette='husl')

sns.swarmplot(x='day', y='total_bill', data=tips, color= 'black')
sns.boxplot(x='day', y='tip', data=tips, palette='husl')

sns.swarmplot(x='day', y='tip', data=tips)
sns.boxplot(x='day', y='total_bill', data=tips, palette='husl')

sns.swarmplot(x='day', y='total_bill', data=tips, color='black')