# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
data.info()
data.describe()
data.columns
data["parental level of education"].unique()

data["average score"] = [data['math score'][each]/3 + data['reading score'][each]/3 + data['writing score'][each]/3 for each in data.index]

data["average score"]
grpA = data[data['race/ethnicity'] == "group A"]

grpB = data[data['race/ethnicity'] == "group B"]

grpC = data[data['race/ethnicity'] == "group C"]

grpD = data[data['race/ethnicity'] == "group D"]

data.head(10)
grpA['math score'].plot(kind='line', alpha=0.5, color='red', label='A-math', grid=True, linewidth=1.5, linestyle='-')

grpA['reading score'].plot(kind='line', alpha=0.5, color='green', label='A-reading', grid=True, linewidth=1.5, linestyle=':')

grpA['writing score'].plot(kind='line', alpha=0.5, color='blue', label='A-writing', grid=True, linewidth=1.5, linestyle='-.')

plt.legend(loc = 'upper right')
grpB['math score'].plot(kind='line', alpha=0.5, color='red', label='B-math', grid=True, linewidth=1.5, linestyle='-')

grpB['reading score'].plot(kind='line', alpha=0.5, color='green', label='B-reading', grid=True, linewidth=1.5, linestyle=':')

grpB['writing score'].plot(kind='line', alpha=0.5, color='blue', label='B-writing', grid=True, linewidth=1.5, linestyle='-.')

plt.legend(loc = 'upper right')
grpC['math score'].plot(kind='line', alpha=0.5, color='red', label='C-math', grid=True, linewidth=1.5, linestyle='-')

grpC['reading score'].plot(kind='line', alpha=0.5, color='green', label='C-reading', grid=True, linewidth=1.5, linestyle=':')

grpC['writing score'].plot(kind='line', alpha=0.5, color='blue', label='C-writing', grid=True, linewidth=1.5, linestyle='-.')

plt.legend(loc = 'upper right')
grpD['math score'].plot(kind='line', alpha=0.5, color='red', label='D-math', grid=True, linewidth=1.5, linestyle='-')

grpD['reading score'].plot(kind='line', alpha=0.5, color='green', label='D-reading', grid=True, linewidth=1.5, linestyle=':')

grpD['writing score'].plot(kind='line', alpha=0.5, color='blue', label='D-writing', grid=True, linewidth=1.5, linestyle='-.')

plt.legend(loc = 'upper right')
course1 = data[data['test preparation course'] == 'completed']

course2 = data[data['test preparation course'] == 'none']
course1['average score'].plot(kind='hist',bins=50,figsize = (12,12))

plt.show()
course2['average score'].plot(kind='hist',bins=50,figsize = (12,12))

plt.show()
data.plot(kind='scatter', x='reading score', y='writing score',alpha = 0.5,color = 'red')

plt.xlabel('Reading')              

plt.ylabel('Writing')

plt.title('Reading Writing Scatter Plot')
data.corr()
f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
sns.countplot(x="parental level of education", data = data, palette="muted")

plt.show()