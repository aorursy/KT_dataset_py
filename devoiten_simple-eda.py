# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import seaborn as sb

import matplotlib.pyplot as plt
data = pd.read_csv('/kaggle/input/health-facilities-gh/health-facilities-gh.csv')
data.head(2)
data.dtypes
data.isnull().sum()
#Plot ownership distribution

plt.figure(figsize=(15,8))

sb.countplot(data['Ownership'], order = data['Ownership'].value_counts(ascending=False).index)

plt.xticks(rotation=80)

sb.despine()
#Synchronizing the ownership variable. Quickest way is just to decapitalize all variable values 

#and drop rows with incorrect values in column

data['Ownership'] = data['Ownership'].str.lower()

data['Ownership'] = data['Ownership'].replace('muslim', 'islamic')



data = data[(data.Ownership != 'clinic')]

data = data[(data.Ownership != 'ngo')]

data = data[(data.Ownership != 'muslim')]

data = data[(data.Ownership != 'maternity home')]

data = data[(data.Ownership != 'mission')]
#Replot distribution

plt.figure(figsize=(15,8))

sb.countplot(data['Ownership'], order = data['Ownership'].value_counts(ascending=False).index)

plt.xticks(rotation=80)

sb.despine()
#Health facilities by type

data['Type'].value_counts()
#Distribution of health facilities by type and ownership

plt.figure(figsize=(15,8))

sb.heatmap(data.groupby('Ownership')['Type'].value_counts(normalize=True).unstack(0), cmap='summer', annot=True)
#Distribution of the ownership by region

plt.figure(figsize=(15,8))

sb.heatmap(data.groupby('Ownership')['Region'].value_counts(normalize=True).unstack(0), cmap='summer', annot=True)