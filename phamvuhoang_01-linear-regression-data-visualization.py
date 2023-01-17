import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data_path = '/kaggle/input/housingprice/ex1data2.txt'



data = pd.read_csv(data_path, sep = ',', header = None)

data.columns = ['Living Area', 'Bedrooms', 'Price']
# Print out first 5 rows to get the imagination of data

data.head()
# Is there any missing data or not?

data.isnull().values.any()
data[['Bedrooms']].plot(kind = 'hist', bins = [0, 1, 2, 3, 4, 5, 6], rwidth = 0.8)
data[['Living Area']].plot(kind = 'hist', bins = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000], rwidth = 0.5)
#data[[ABC]].plot(kind = 'hist', bins = [xxx, yyy, ....], rwidth = 0.8)
data.hist(rwidth = 1)
data.groupby('Bedrooms')['Price'].nunique().plot(kind = 'bar')
data.plot(kind='scatter', x = 'Bedrooms', y = 'Price', color = 'green')
data.plot(kind = 'scatter', x = 'Living Area', y = 'Bedrooms', color = 'blue')
#data.plot(kind='scatter', x = AAA, y = BBB, color = CCC)
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
X0 = data['Bedrooms']

Y0 = data['Living Area']

Z0 = data['Price']



_3d_figure = plt.figure(figsize = (15, 10)).gca(projection = '3d')

_3d_figure.plot(X0, Y0, Z0)



plt.show()
X1 = range(data.shape[1])

Y1 = range(data.shape[0])

X1, Y1 = np.meshgrid(X1, Y1)



_3d_figure = plt.figure(figsize = (15, 10)).gca(projection = '3d')

_3d_figure.plot_wireframe(X1, Y1, data)



plt.show()
from sklearn.linear_model import LinearRegression
X = data['Bedrooms'].values.reshape(-1, 1)

Y = data['Price'].values.reshape(-1, 1)



# Visualize the data

plt.scatter(X, Y)



# Train the model

model = LinearRegression()

model.fit(X, Y)



# Predict with the same input data

Y_pred = model.predict(X)



# Draw the linear regression model

plt.plot(X, Y_pred, color = 'red')



plt.show()
#X = data[AAA].values.reshape(-1, 1)

#Y = data[BBB].values.reshape(-1, 1)



# Visualize the data

#plt.scatter(X, Y)



# Train the model

#model = LinearRegression()

#model.fit(X, Y)



# Predict with the same input data

#Y_pred = model.predict(X)



# Draw the linear regression model

#plt.plot(X, Y_pred, color = 'green')



#plt.show()
#X_test = [[AAA]]

#Y_test = model.predict(X_test)



#print(Y_test)