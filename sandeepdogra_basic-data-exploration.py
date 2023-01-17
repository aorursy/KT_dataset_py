# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#read the CSV and get the data
df = pd.read_csv('/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv', delimiter=',')
df.head()
#get th description of the data
df.describe()
df.tail()
df.rename(columns={'Unnamed: 0':'Index'}, inplace= True)
df.head()
df.columns
df.dtypes #getdata type fo the columns
df.corr() #get the coorelation between the variables
#get regresssion plot b/w year and price
sns.regplot(x="year", y="price", data=df)
plt.ylim(0,)
##get regresssion plot b/w mileage and price
sns.regplot(x="mileage", y="price", data=df)
plt.ylim(0,)
#box ploting for categorical variables
sns.boxplot(x="brand", y="price", data=df)
# grouping results
df_gptest = df[['brand','model','price']]
grouped_test1 = df_gptest.groupby(['brand','model'],as_index=False).mean()
grouped_test1
#create a pivot
grouped_pivot = grouped_test1.pivot(index='brand',columns='model')
grouped_pivot
grouped_pivot.fillna(0)
#use the grouped results
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()

