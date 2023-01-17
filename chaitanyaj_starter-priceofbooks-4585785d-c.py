from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Import dataset
dataset = pd.read_csv('/kaggle/input/priceofbooks-csv/Data_Train.csv')
#preparing data 
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#splitting the data into trainingset and testset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
#prepare categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder()
X = oh.fit_transform(X_train[:,[0, 1,3, 4, 6, 7]]).toarray()

#fit algorithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#X = np.array(X.reshape(X.shape[0]))

regressor.fit(X, y_train)
y_pred = regressor.predict(X)
print(y_pred)
print(y_train)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
mean_squared_error(y_train, y_pred)
#plot the result 
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X))