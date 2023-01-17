# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/automobile-data/automobile.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#uploaded data from my local machine to kaggle kernal and reading the csv file.

df = pd.read_csv('../input/automobile.csv')
df.head()
df.tail()
#checking all the columns 

df.columns
df.info()
df.describe()
df['drive-wheels'].value_counts()
#we will check how the number of cylinder in a vehical effects the price of vehical

sns.boxplot(x = 'num-of-cylinders', y = 'price',data = df)
#let's check how the "drive-wheel" an effect on price.

sns.boxplot(x= 'drive-wheels', y = 'price', data= df)
plt.scatter(df['engine-size'], df['price'])

plt.xlabel('Engine size')

plt.ylabel('Price')

plt.show()
counts, bin_edges = np.histogram(df['peak-rpm'])

df['peak-rpm'].plot(kind ='hist', xticks=bin_edges)

plt.xlabel('Peak rpm value')

plt.ylabel('numbers of cars')

plt.grid()

plt.show()
df_group_average = df[['num-of-doors', 'body-style', 'price']]

price = df_group_average.groupby(['num-of-doors', 'body-style'], as_index = False).mean()

print(price)
#more detailed 

price.pivot(index = 'body-style', columns = 'num-of-doors')
#checking null values.

df.isnull()
#this may seems to a convient way to check for missing values.

#we can use heatmap to visualize missing values in data.

sns.heatmap(df.isnull())