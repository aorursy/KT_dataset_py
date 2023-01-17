# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
s1 = pd.Series([1,3,6,np.nan,44,1])

print(type(s1))

s1
s2 = pd.Series([1,2,3], index=['a','b','c'])

print(type(s2))

s2
df = pd.DataFrame(np.random.randn(6,4),columns=['a','b','c','d'])

print(type(df))

df
from sklearn import datasets

iris = datasets.load_iris()

iris
iris.keys()
print(iris['data'])
iris_data1 = pd.DataFrame(iris['data'],columns=iris['feature_names'])

iris_data1
iris_data2 = pd.DataFrame(iris['target'],columns=['target'])

iris_data2
iris_data = pd.concat([iris_data1,iris_data2],axis=1)

iris_data
iris_data.info()
iris_data.describe()
iris_data['sepal length (cm)'].head(5)
iris_data[['sepal length (cm)','sepal width (cm)']].head(5)
iris_data[5:10][['sepal length (cm)','sepal width (cm)']]
iris_data.loc[5:10,['sepal length (cm)','sepal width (cm)']]
iris_data[iris_data['sepal length (cm)']>2]
iris_data.groupby(by='target').sum()
iris_data.groupby(by='target').mean()