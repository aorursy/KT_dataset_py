import numpy as np # for numeric

import pandas as pd # It is for data structure

import matplotlib.pyplot as plt #



dataset = pd.read_csv("../input/Housing_Data.csv")

x = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 1].values



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/5, random_state = 0)



from sklearn.linear_model import LinearRegression

regression = LinearRegression() # regression is an object

regression.fit(x_train, y_train)



# predicting the test set 

y_pred = regression.predict(x_test)



plt.scatter(x_train, y_train, color = 'red')

plt.plot(x_train, regression.predict(x_train), color = 'blue')

plt.title('Housing Data (Training set)')

plt.xlabel('Area')

plt.ylabel('Price')

plt.show()