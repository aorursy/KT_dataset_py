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
import csv



d=[]



with open('/kaggle/input/weatherww2/Summary of Weather.csv') as csv_file:

    csv_reader = csv.reader(csv_file, delimiter=',')

    line_count = 0

    for row in csv_reader:

        if line_count == 0:

            head = row

            line_count += 1

        else:

            d.append(row)

            line_count += 1

    print(f'Processed {line_count} lines.')
data = np.array(d)

data[1]
head
y = data[:,4]

np.shape(y)

y = y.reshape(-1, 1)

y = y.astype(np.float64)
from sklearn import preprocessing

X = data[:,5]

np.shape(X)

X = X.reshape(-1, 1)

X = X.astype(np.float64)

X = preprocessing.scale(X)
import matplotlib.pyplot as plt

fig = plt.figure()

ax = fig.add_axes([0,0,3,3]) 

ax.scatter(X, y, color = "red")

plt.show
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train,y_train)
print('train accuracy: ',reg.score(X_train,y_train)*100)

print('test accuracy:  ',reg.score(X_test,y_test)*100)
reg.predict([[-2]])
fig = plt.figure()

ax = fig.add_axes([0,0,3,3]) 

ax.scatter(X_train, y_train, color = "red")

ax.plot(X_train, reg.predict(X_train), color = "g")

plt.show