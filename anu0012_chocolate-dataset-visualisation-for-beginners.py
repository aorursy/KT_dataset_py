# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import os

import pandas as pd

import sys

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(color_codes=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/flavors_of_cacao.csv')
df.head()
df.isnull().sum(axis=0)
df.dtypes
sns.distplot(df['REF'])
sns.distplot(df['Review\nDate'])
sns.distplot(df['Rating'])
#scatter plot 

var = 'REF'

data = pd.concat([df['Rating'], df[var]], axis=1)

data.plot.scatter(x=var, y='Rating');
#scatter plot 

var = 'Review\nDate'

data = pd.concat([df['Rating'], df[var]], axis=1)

data.plot.scatter(x=var, y='Rating');
df['Company\nLocation'].value_counts()
df['Company\nLocation'].value_counts().head(10).plot.bar()
df['Cocoa\nPercent'].value_counts()
df['Cocoa\nPercent'].value_counts().head(10).plot.bar()
df[df['Cocoa\nPercent'] == '100%']
df['Rating'].value_counts().sort_index().plot.bar()
df[df['Rating'] == 5.0]