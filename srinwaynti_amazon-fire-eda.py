# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        path=os.path.join(dirname, filename)

        print(path)



# Any results you write to the current directory are saved as output.
amazon=pd.read_csv(path,encoding = "ISO-8859-1")

amazon.head()
amazon.info()
amazon.isnull().sum() #Chedking for null values
amazon['date'].describe() #Total count of date is 6454, but the unique count is 20

amazon['date'].value_counts()

#The number of entries is distributed across almost all the years, except 2017.

#This might be because the data was taken at the middle of the year 2017,and so all the data is not filled.
amazon['year'].value_counts()

amazon.drop('date',axis=1,inplace=True)
plt.figure(figsize=(50,40))

sns.barplot(x='month',y='number',data=amazon)

plt.figure(figsize=(50,40))

sns.barplot(x='year',y='number',data=amazon)
sns.barplot(x='state',y='number',data=amazon)