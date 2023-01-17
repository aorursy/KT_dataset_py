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
from pandas import DataFrame

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression



vital_data = pd.read_csv('../input/weight-height.csv')

vital_data.head()
vital_data.describe()
X = DataFrame(vital_data,columns=['Height'])

y = DataFrame(vital_data,columns=['Weight'])



plt.scatter(X,y,alpha=0.3)

plt.title('Vital Stats')

plt.xlabel('Height in inches')

plt.ylabel('Weight in kgs')

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
#Create the Model

regression = LinearRegression()

regression.fit(X_train,y_train)



#Predict the Weight values on the basis of the Test Height values

y_Pred = regression.predict(X_test)
plt.scatter(X_train,y_train,alpha=0.3)

plt.plot(X_train,regression.predict(X_train),color='red',linewidth=3)

plt.title('Vital Stats')

plt.xlabel('Height in inches')

plt.ylabel('Weight in kgs')

plt.show()
# Predicting the Test set results

print('Coefficients: ', regression.coef_)

# The mean squared error

print("Mean squared error: %.2f" % np.mean((regression.predict(X_test) - y_test) ** 2))

# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % regression.score(X_test, y_test))