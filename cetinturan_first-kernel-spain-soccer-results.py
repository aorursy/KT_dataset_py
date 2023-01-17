# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
matches = pd.read_csv('../input/FMEL_Dataset.csv')
matches.info()
matches.corr()
f,ax = plt.subplots(figsize=(13, 13))

sns.heatmap(matches.corr(), annot=True, linewidths=.3, fmt= '.2f',ax=ax)

plt.show()
matches.head(8)
matches.columns
matches.localGoals.plot(kind = 'line', color = 'g',label = 'localGoals',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

matches.visitorGoals.plot(color = 'r',label = 'visitorGoals',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
matches.plot(kind='scatter', x='localGoals', y='visitorGoals',alpha = 0.5,color = 'red')

plt.xlabel('Local Goals')              

plt.ylabel('Visitor Goals')

plt.title('Local vs Visitor Goals')            
matches.localGoals.plot(kind = 'hist',bins = 20,figsize = (10,10))

plt.show()
matches.visitorGoals.plot(kind = 'hist',bins = 20,figsize = (10,10))

plt.show()
matches[np.logical_and(matches['localGoals']>2, matches['visitorGoals']>1 )]
matches['totalScore'] = matches['localGoals'] + matches['visitorGoals']
matches.head(10)
matches.corr()
f,ax = plt.subplots(figsize=(13, 13))

sns.heatmap(matches.corr(), annot=True, linewidths=.3, fmt= '.2f',ax=ax)

plt.show()
matches['underOver'] = [ 0 if i<3 else 1 for i in matches.totalScore]
matches.head(10)
matches.corr()
matches.info()
matches.corr()
f,ax = plt.subplots(figsize=(13, 13))

sns.heatmap(matches.corr(), annot=True, linewidths=.3, fmt= '.2f',ax=ax)

plt.show()
matches['localTeam'].value_counts(dropna=False)
matches['visitorTeam'].value_counts(dropna=False)
matches.describe()
matches.boxplot(column='localGoals', by='underOver')

plt.show()
matches.dtypes