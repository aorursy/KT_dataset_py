#Linear Regressions

#STEP 1 PRE PROCESSING DATA

#IMPORT LIBRARARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORT AND CLEAN DATASET
train_dataset = pd.read_csv('../input/train.csv')
train_dataset = train_dataset[train_dataset.x<100]
X_train = train_dataset.iloc[:,:-1].values
Y_train = train_dataset.iloc[:,1].values

test_dataset = pd.read_csv('../input/test.csv')
test_dataset = test_dataset[test_dataset.x<100]
X_test = test_dataset.iloc[:,:-1].values
Y_test = test_dataset.iloc[:,1].values

#STEP 2 BUILD MODEL

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#STEP 3 RUN THE MODEL ON TEST DATA
Y_pred = regressor.predict(X_test)

#STEP 4 VISUALISATION
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('X Vs. Y(Training Set)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('X Vs. Y(Test Set)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
