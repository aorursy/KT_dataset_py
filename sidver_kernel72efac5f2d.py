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
df = pd.read_csv('/kaggle/input/heights-and-weights/data.csv')
df.info()
%matplotlib inline

import matplotlib.pyplot as plt
plt.scatter(x=df['Height'],y=df['Weight'],color='red')
X = df.iloc[:, :-1].values

y = df.iloc[:, 1:].values
from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train , y_train)
plt.scatter(x=df['Height'],y=df['Weight'],color='red')

plt.plot(X_train,regressor.predict(X_train),color='green')

plt.title('Height vs Weight')

plt.xlabel('Height')

plt.ylabel('Weight')
regressor.coef_
regressor.intercept_
plt.scatter(X_test,y_test,color='red')

plt.plot(X_train,regressor.predict(X_train),color='green')

plt.title('Height vs Weight(Test Set)')

plt.xlabel('Height')

plt.ylabel('Weight')
from sklearn.metrics import r2_score

y_pred = regressor.predict(X_test)

r2_score(y_test,y_pred)