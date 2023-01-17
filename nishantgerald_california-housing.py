# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
file='../input/housing.csv'

df=pd.read_csv(file)
df
df.corr(method ='pearson')
df.plot(kind='scatter',x='median_income',y='median_house_value')

sns.regplot(x="median_income", y="median_house_value", data=df,color='r')
df.describe()