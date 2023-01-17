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
from sklearn.model_selection import train_test_split

data = pd.read_csv('/kaggle/input/process-attributes-netbsd/process_attributes_netbsd.csv')

for col in data.columns:

    print(col)
#data.head()

y = data[data.columns[-1]]

y.head()
x = data.drop('TAT', axis=1)

x.head()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
x_train.head()

x_train.shape
x_test.head()

x_test.shape
from sklearn import linear_model

import matplotlib.pyplot as plt

lr = linear_model.LinearRegression()

model = lr.fit(x_train, y_train)

print('Train Score:', model.score(x_test, y_test))
print ("Skew is:", data.TAT.skew())

plt.hist(data.TAT, color='blue')

plt.show()
target=np.log(data.TAT)

print("Skew after log transform:", target.skew())

plt.hist(target, color='blue')

plt.show()
predictions = model.predict(x_test)
from sklearn.metrics import mean_squared_error

print('Error', mean_squared_error(y_test, predictions))
act_values = y_test

plt.scatter(predictions, act_values, alpha=.5, color='blue')

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.title('Linear Regression')

plt.show()
# final_pred = predictions

# print ("Original predictions are: \n", predictions[:5], "\n")

# print ("Final predictions are: \n", final_pred[:5])



y_test.head()
predictions
from sklearn.metrics import r2_score

r2_score(y_test, predictions)