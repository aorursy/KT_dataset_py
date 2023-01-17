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
df = pd.read_csv("../input/titanicdataset-traincsv/train.csv")
df.head()
df.tail()
df.info()
df.describe()
includer_list = ['object','float64','int64']

df.describe(include = includer_list)
df[0:10]
df.iloc[1]
#Seeing the names of the columns of our dataframe:

cols = df.columns

for item in cols:

    print(item)
#Eliminating the columns Parch and SibSp

df = df.drop(['Parch', 'SibSp'], axis = 1)
df.head()
#Seeing the values of the indexes of our dataframe:

df.index.values
#Setting the indexes of our dataframe to be the names of the passengers:

df = df.set_index('Name')
#Seeing the values of the indexes of our dataframe:

df.index.values
df.isnull().sum()
#Seeing how many passengers are male and female

df.Sex.value_counts()