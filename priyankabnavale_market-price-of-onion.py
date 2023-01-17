# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/market-price-of-onion-2020/Onion Prices 2020.csv")
df.head()
df.shape
df.info()
sns.countplot(x='variety',data=df)
sns.barplot(df['variety'],df['modal_price'])
df=pd.get_dummies(df)
df.head()
x =df.loc[:,df.columns!='modal_price']
x.shape
y= df['modal_price']
y.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.head()
x_train.shape
y_train.head()
y_train.shape
x_test.shape
y_test.shape
from sklearn.linear_model import LinearRegression
model=LinearRegression().fit(x_train,y_train)
model.score(x_test,y_test)*100
model.predict(x_test)
y_pred=model.predict(x_test)