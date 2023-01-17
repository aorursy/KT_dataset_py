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
import pandas as pd

import seaborn as sn

import numpy as np

import matplotlib.pyplot as plt
df=pd.read_csv("/kaggle/input/class123/diamonds.csv")
df.head()
df.info()
df.isnull().sum()
df.describe()
df.price.hist()
df.carat.hist()
df.depth.hist()
df.table.hist()
df.price.hist(bins=10

             )
sn.distplot(df.price)
sn.distplot(df.price,hist=False)
df.columns
sn.barplot(x=df.color,y=df.price)
from numpy import median
sn.barplot(x=df.color,y=df.price,estimator=median)
sn.barplot(x=df.cut,y=df.price)
sn.barplot(x=df.clarity,y=df.price)
sn.boxplot(x=df.color,y=df.price)
sn.boxplot(x=df.cut,y=df.price)
sn.boxplot(x=df.clarity,y=df.price)
sn.heatmap(df.corr().round(2),annot=True)
sn.set_style("ticks")

sn.set(rc={"figure.figsize":(10,8)})

sn.heatmap(df.corr().round(2),annot=True)
sn.heatmap(df[['carat','depth','price']].corr(),annot=True)
sn.pairplot(df)
plt.scatter(x=df.price,y=df.carat)