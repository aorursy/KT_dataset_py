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
import matplotlib.pyplot as plt
df_train = pd.read_csv("../input/random-linear-regression/train.csv")

df_test = pd.read_csv("../input/random-linear-regression/test.csv")
df_train.sample(5)
df_train.tail(5)
df_train.isnull().any()
df_train = df_train.dropna()
df_train.isnull().any()
df_train.info()
X_train = df_train.iloc[:,:-1]    # X_train = df_train['x']

y_train = df_train.iloc[:,-1]     # y_train = df_train['y']
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)
X_test = df_test.iloc[:,:-1]    # X_test = df_test['x']

y_test = df_test.iloc[:,-1]     # y_test = df_test['y']
pred = reg.predict(X_test)
plt.scatter(X_train,y_train,color='red')

plt.plot(X_test,pred,color='green')  # To check the correctness of prediction

plt.show()
#Our model is perfectly fitted