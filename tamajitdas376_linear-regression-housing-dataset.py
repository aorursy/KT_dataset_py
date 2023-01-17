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
import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('../input/USA_Housing.csv')

df.head()
df.info()
df.describe()

sns.pairplot(df)
df.columns
X=df['Avg. Area Number of Rooms']

y=df['Price']

print(X)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X.values.reshape(-1,1),y,test_size=0.25,random_state=101)
from sklearn.linear_model.stochastic_gradient import SGDRegressor
lm=SGDRegressor(loss="squared_loss",max_iter=20000,penalty="none")
lm.fit(X_train,y_train)
pred_train=lm.predict(X_train)

pred_test=lm.predict(X_test)
pred_train
y_train
plt.subplot(1,2,1)

plt.title('Train')

plt.scatter(X_train,y_train)

plt.plot(X_train, pred_train)



plt.subplot(1,2,2)

plt.title('Test')

plt.scatter(X_test,y_test)

plt.plot(X_test,pred_test)