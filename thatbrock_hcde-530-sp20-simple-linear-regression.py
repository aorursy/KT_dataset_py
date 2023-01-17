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
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("/kaggle/input/task3-ttc/task3_TTC.csv")
data.head()
X = data.iloc[:, 0].values.reshape(-1,1)
Y = data.iloc[:, 1].values.reshape(-1,1)

plt.figure(figsize=(8,8))
plt.scatter(X, Y, s=30)
plt.title('Task 3 completion time vs. Age')
plt.xlabel('Participant Age')
plt.ylabel('Time to complete Task 3 (in seconds)')
plt.show()
model = LinearRegression()
model.fit(X,Y)
Y_pred = model.predict(X)
plt.figure(figsize=(8,8))
plt.scatter(X, Y, s=30)
plt.plot(X, Y_pred, color = 'red')
plt.title('Task 3 completion time vs. Age')
plt.xlabel('Participant Age')
plt.ylabel('Time to complete Task 3 (in seconds)')
plt.show()
cars = pd.read_csv("../input/original-cars/original cars.csv")
cars.tail()
X = cars.iloc[:, 5].values.reshape(-1, 1)  # values converts it into a numpy array
Y = cars.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
type(X)
model = LinearRegression()  # create an object called `model` for the LinearRegression class
model.fit(X, Y)  # perform linear regression on the model using X and Y from the file we have read in
Y_pred = model.predict(X)  # make predictions

plt.figure(figsize=(8,8))
plt.scatter(X, Y, s=10)
plt.plot(X, Y_pred, color='red')
plt.title('Increased Weight Reduces Fuel Economy')
plt.xlabel('Weight (lbs)')
plt.ylabel('MPG')
plt.show()