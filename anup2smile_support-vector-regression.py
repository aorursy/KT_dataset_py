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
#importing the libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
#Importing the Dataset



Dataset = pd.read_csv('../input/Position_Salaries.csv')
Dataset.head()
#Assigning the Predictor and outcome variables
X = Dataset.iloc[:, 1:2]

y = Dataset.iloc[:, 2:]



print(X.head())

print(y.head())
#Implying Feature scaling in the dataset
from sklearn.preprocessing import StandardScaler



sc_X = StandardScaler()

sc_y = StandardScaler()



X = sc_X.fit_transform(X)

y = sc_y.fit_transform(y)



#Bulding the model with SVR
from sklearn.svm import SVR



regressor = SVR(kernel = 'rbf')



regressor.fit(X, y)

y_pred = sc_y.inverse_transform(regressor.predict(X))



y_pred
y_pred = y_pred.reshape((10,1))
y_pred.shape
Add = np.append(arr= sc_y.inverse_transform(y), values=y_pred, axis = 1)

Add
#Visualising the SVR results
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y), color = 'red')

plt.plot(sc_X.inverse_transform(X), y_pred , color = 'blue')

plt.title('Bluff vs truth')

plt.xlabel('Position')

plt.ylabel('salary')

plt.show()