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




train = pd.read_csv("../input/random-linear-regression/train.csv") 

test = pd.read_csv("../input/random-linear-regression/test.csv") 

train = train.dropna()

test = test.dropna()





from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train = scaler.fit_transform(train)

test = scaler.fit_transform(test)
X_train = np.array(train[:, :-1])

y_train = np.array(train[:, 1])

X_test = np.array(test[:, :-1])

y_test = np.array(test[:, 1])




import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor = regressor.fit(X_train, y_train)

Y_pred = regressor.predict(X_test)

# plt.scatter(X_train , y_train, color = 'red')

# plt.plot(X_train , regressor.predict(X_test), color ='blue')

print("model score : ", regressor.score(X_test, y_test))


