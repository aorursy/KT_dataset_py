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

import numpy as np

import matplotlib.pyplot as plt

from sklearn import tree
def converter(x):

    if x=='<1H OCEAN':

        return 0

    if x=='INLAND':

        return 1

    if x=='NEAR OCEAN':

        return 2

    if x=='NEAR BAY':

        return 3

    else:

        return 4  
import pandas as pd

df = pd.read_csv("../input/california-housing-prices/housing.csv",converters={'ocean_proximity':converter})
df.tail()
df['ocean_proximity'].value_counts()
df.head()

df1=df.head(100)

df1.round(2)
df.corr()
df.plot(kind='scatter',x='housing_median_age',y='median_income',alpha=0.1)
x=df1.iloc[:,:-3]

x['ocean_proximity']=df1[['ocean_proximity']]

x['median_income']=df1['median_income']

y=df1['median_house_value']

x
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=25,max_depth=5)

clf.fit(x, y)
ypredict=clf.predict(xtest)
from sklearn.metrics import explained_variance_score
result=explained_variance_score(ytest,ypredict,multioutput='uniform_average')
result