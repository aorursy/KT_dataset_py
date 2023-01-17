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
df =pd.read_csv('/kaggle/input/insurance/insurance.csv')
df.head()
import seaborn as sns
sns.scatterplot(x = df.age,y = df.charges,hue=df.smoker,style=df.sex)
df.isnull().sum()
df.info()
df

one_hot=pd.get_dummies(df['sex'])

df = df.drop('sex',axis=1)

df = df.join(one_hot)

df
one_hot = pd.get_dummies(df['region'])

df = df.drop('region',axis=1)

df = df.join(one_hot)
one_hot=pd.get_dummies(df['smoker'])

df =df.drop('smoker',axis=1)

df = df.join(one_hot)

df
df['charges'] = df['charges'].astype(int)
from sklearn.model_selection import train_test_split

train, test = train_test_split(df,test_size = 0.1,random_state = 1)
def data_split(df):

    x=df.drop('charges',axis=1)

    y=df['charges']

    return x,y

x_train,y_train=data_split(train)

x_test,y_test = data_split(test)
from sklearn.ensemble import RandomForestRegressor



regress = RandomForestRegressor()

regress.fit(x_train , y_train)



reg_train = regress.score(x_train , y_train)

reg_test = regress.score(x_test , y_test)



print(reg_train)

print(reg_test)