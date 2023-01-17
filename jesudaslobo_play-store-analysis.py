# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_frame = pd.read_csv("../input/GooglePlayStoreBEFORE.csv")

data_frame.head(10)
data_frame.shape
data_frame.describe()
data_frame.boxplot()
data_frame.boxplot(column=['Rating'], return_type='axes');
data_frame.hist()
data_frame.info()
data_frame.isnull()
data_frame.isnull().sum()
data_frame[data_frame.Reviews > 100000]
threshhold = len(data_frame) * 0.1

data_frame.dropna(thresh = threshhold, axis = 1, inplace=True)

data_frame.shape
data_frame['Price']
data_frame['Price'] = data_frame['Price'].apply(lambda x : str(x).replace('$','') if '$' in str(x) else str(x))

#data_frame['Price'].str.replace('$','')

data_frame['Price'] = data_frame['Price'].apply(lambda x : float(x))
data_frame[data_frame.Price > 0]
# Data Visualization

grp = data_frame.groupby('Category')

x = grp['Rating'].agg(np.mean)

y = grp['Price'].agg(np.sum)

z = grp['Reviews'].agg(np.mean)
plt.plot(x)
plt.plot(x,'ro')
plt.figure(figsize=(12,5))

plt.plot(z,'ro')

plt.xticks(rotation=90)

plt.show()