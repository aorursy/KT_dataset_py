# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/all-crypto-currencies/crypto-markets.csv")

df.head(4)

df.shape
crypto=df.dropna()

crypto.shape


from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,r2_score

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.model_selection import cross_val_score
x = crypto.iloc[:,5:8].values

y = crypto.iloc[:,8].values

X_train,  X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,

                                                    random_state=16)


model = LinearRegression()

model.fit (X_train,y_train)
m = model.coef_

c = model.intercept_

close_price = (m*X_train+c)

close_price_predict = model.predict(close_price)

close_price_predict1 = (m*X_test+c)

close_price_test_predict = model.predict(close_price_predict1)

close_price_predict, close_price_test_predict