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
series1=np.arange(8)

series1
series1[2]
np.random.seed(20)

df=pd.DataFrame(np.random.rand(36).reshape((6,6)), index=['r1','r2','r3','r4','r5','r6'], 

                columns=['col1','col2','col3','col4','col5','col6'])
df
df.loc['r1':'r3','col1':'col3']
df.iloc[3:5,2:6]
df.iloc[0,5]
df.iloc[:,0:2]
df.iloc[4,0]
df>0.5
df1=df[df>0.3]

df1
df1.isnull().sum()
df1.fillna(0)
df2=df1[df1<0.8]

df2
df2['col1'].replace(np.nan,0.5, inplace=True)
df2.drop(['col4'], axis=1, inplace=True)
df2
df2.apply(lambda x : x.fillna(x.mean()), axis=1)
data=pd.DataFrame({'name': ['x','y', 'z', 'x'], 'purchase': [1,2,3,1]})

data
data.duplicated()
data.drop_duplicates()
data1=pd.DataFrame(np.random.rand(36).reshape((6,6)))

data1
data2=pd.DataFrame(np.random.rand(15).reshape((5,3)))

data2
pd.concat([data1, data2], axis=1)
pd.concat([data1, data2], axis=0)
data1.drop([0,2])
data1.drop([0,2], axis=1)
from pandas import Series, DataFrame
series_add=Series(np.arange(6))

series_add.name="added_variable"

series_add
add_data_frame=DataFrame.join(data2,series_add)

add_data_frame
add=add_data_frame.append(add_data_frame, ignore_index=False)

add
add=add_data_frame.append(add_data_frame, ignore_index=True)

add
data2.sort_values(by=[1], ascending=False, inplace=True)

data2
cars=pd.read_csv("../input/mt1cars.csv")

cars.head()
cars_grouped=cars.groupby(cars["cyl"])

cars_grouped.mean()
cars_gear=cars.groupby(cars["gear"])

cars_gear.mean()
import matplotlib.pyplot as plt

import seaborn as sns
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(cars.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax, cmap='RdYlGn')

plt.show()

sns.pairplot(data=cars)