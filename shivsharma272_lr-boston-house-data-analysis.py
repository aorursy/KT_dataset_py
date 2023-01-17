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
ds = pd.read_csv("/kaggle/input/boston-housing-dataset/HousingData.csv")
ds.head(3)
ds.tail()
ds
ds.describe()
ds.info()
ds.isnull().sum()
ds=ds.dropna()     #drop all rows that have any NaN values
ds.isnull().sum()
x=ds.drop('TAX',axis=1)

y=ds['TAX']
x
y
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=.22,random_state=42)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()



import warnings

warnings.filterwarnings('ignore')
lr.fit(train_x,train_y)

lr.score(train_x,train_y)