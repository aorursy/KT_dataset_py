# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import matplotlib.pyplot as plt

import seaborn as sns

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
df_train=pd.read_csv('/kaggle/input/random-linear-regression/train.csv')

df_test=pd.read_csv('/kaggle/input/random-linear-regression/test.csv')
df_train.head()

df_train.info()
# from the above code we see that y has 1 null value so we will either drop the row or replace the 

# value with mean of colum 'y'

# Here we will drop the row since we have large dataset 



df_train.dropna(inplace=True)

df_train.info()
df_test.info()
plt.scatter(df_train['x'],df_train['y'],color='g',marker='.')

plt.xlabel('X')

plt.ylabel('Y')

plt.show()
X_train=df_train[['x']]

y_train=df_train[['y']]

X_test=df_test[['x']]

y_test=df_test[['y']]
from sklearn.linear_model import LinearRegression

regr=LinearRegression()

regr.fit(X_train,y_train)
from sklearn.metrics import r2_score

yhat=regr.predict(X_test)

print("The score of model is :",r2_score(yhat,y_test))
plt.figure(figsize=(10,8))

plt.scatter(X_test,y_test,color='g',marker='.',label='Test Points')

plt.plot(X_test,yhat,color='blue',label='Regression Line')

plt.xlabel('X')

plt.ylabel('Y')

plt.legend(loc='lower right')

plt.title('X vs Y')

plt.show()