# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from sklearn import linear_model, metrics

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dat = pd.read_csv("/kaggle/input/dataset.csv", sep=";", decimal=",")

dat
x = np.array(dat['Years of Experience'])

x
y = np.array(dat['Salary'])

y
plt.scatter(x,y)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 4)

x_train = x_train.reshape(-1,1)

y_train = y_train.reshape(-1,1)

x_test = x_test.reshape(-1,1)

y_test = y_test.reshape(-1,1)
reg = linear_model.LinearRegression(normalize=True)

reg.fit(x_train,y_train)
pred = reg.predict(x_test)

print(pred[6])
mean_squared_error=metrics.mean_squared_error(y_test,pred)

print('Sqaured mean error', round(np.sqrt(mean_squared_error),2))

print('R squared training',round(reg.score(x_train,y_train),3))

print('R sqaured testing',round(reg.score(x_test,y_test),3) )

print('intercept',reg.intercept_)

print('coefficient',reg.coef_)
plt.scatter(x_test,y_test)

plt.plot(x_test,pred, color="Red")
print(np.corrcoef(x,y))