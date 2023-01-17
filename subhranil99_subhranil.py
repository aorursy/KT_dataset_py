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
import matplotlib.pyplot as plt
fields = ['T_degC','Salnty']



# Taking only the first 500 values

dataset = pd.read_csv("/kaggle/input/calcofi/bottle.csv" , usecols = fields, dtype = {'T_degC' : float, 'Salnty' : float})

X = dataset.iloc[:500,0].values

y = dataset.iloc[:500,1].values
X = np.reshape(X,(-1,1))
# Removing Missing Values 

from sklearn.preprocessing import Imputer

imputer_X = Imputer(missing_values = 'NaN', strategy = 'mean' , axis = 0)

X = imputer_X.fit_transform(X)

imputer_y = Imputer(missing_values = 'NaN', strategy = 'mean' , axis = 0)

y = imputer_y.fit_transform(np.reshape(y,(-1,1)))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#Building the Regressor

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,y_train)
# Predicting the Test Set Results

y_pred = regressor.predict(y_test)
# Visualising Training Set

plt.scatter(X_train,y_train, color = "red")

plt.plot(X_train, regressor.predict(X_train) , color = "blue")

plt.title("Oceanographic Data")

plt.xlabel("Temperature in degree Celcius")

plt.ylabel("Salinity")
# Visualising the Test Set

plt.scatter(X_test,y_test, color = "red")

plt.plot(X_test, regressor.predict(X_test) , color = "blue")

plt.title("Oceanographic Data")

plt.xlabel("Temperature in degree Celcius")

plt.ylabel("Salinity")