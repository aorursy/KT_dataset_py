#importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

#importing data

df = pd.read_excel('../input/slr02.xls')

df.head()
# to know if it's a linear regression problem,simply plot the scatter plot between the dependent and independent variables

plt.scatter(df['X'],df['Y'])

plt.xlabel('chirps/sec')

plt.ylabel('temperature in farheneit')

plt.show()

# convert the values to numpy for processing

X = df.iloc[:,:-1].values

Y = df.iloc[:,1].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4,random_state = 51)
#import Linear Regression model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)



# Visualising the Training set results

plt.scatter(X_train, y_train, color = 'red')

plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.title('No of chirps/sec VS temperature')

plt.ylabel("Temperature('farheneit')")

plt.xlabel('Chirps/sec')

plt.show()
# Visualising the Test set results

plt.scatter(X_test, y_test, color = 'orange')

plt.plot(X_train, regressor.predict(X_train), color = 'black')

plt.title('No of chirps/sec VS temperature')

plt.ylabel("Temperature('farheneit')")

plt.xlabel('Chirps/sec')

plt.show()
accuracy = regressor.score(X_test,y_test)

print("%.2f"%(accuracy*(100)))
#We Should Alwyas have a great quantity of data to develop a more accurate model
r = regressor.predict([[17.2]])
print(r)