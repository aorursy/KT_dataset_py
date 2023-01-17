# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv')
df.info()
X = df.iloc[:,:-1].values

Y = df.iloc[:, -1].values
from sklearn.model_selection import train_test_split as split

X_train,X_test,Y_train,Y_test = split(X, Y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression

lin = LinearRegression()

lin.fit(X_train,Y_train)
Y_pred = lin.predict(X_test)
Y_pred
print('Y_test  ------  Y_pred')

for i,j in zip(Y_test,Y_pred):

    print(f'{i} ---- {j}\n')
plt.scatter(X_train,Y_train,color = 'Red')

plt.plot(X_train,lin.predict(X_train),color = 'blue')
plt.scatter(X_test,Y_test,color = 'Red')

plt.plot(X_train,lin.predict(X_train),color = 'magenta')
from sklearn.metrics import r2_score

# to calculate the accuracy of the model out of 1 or 100 %

print(f'accuracy is {r2_score(Y_test, Y_pred)} %')