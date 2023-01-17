import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import keras

from keras.models import Sequential

from keras.layers import Dense

from sklearn.metrics import r2_score,mean_squared_error

import time

import math
dataset = pd.read_csv('../input/random-salary-data-of-employes-age-wise/Salary_Data.csv')
dataset.shape
dataset.isnull().sum()
dataset.head()
X = dataset.iloc[:,0:1].values 

y = dataset.iloc[:,1].values

print("X Shape: ",X.shape,'\ny Shape :',y.shape)
plt.figure(figsize=(20,10))

plt.scatter(X,y, color='red')

plt.title('Salary vs Exp Scatterplot')

plt.xlabel('years of exp')

plt.ylabel('Salary')

plt.show()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=0)
start_1 = time.time()

regressor = LinearRegression()

regressor.fit(X_train,y_train)

end_1 = time.time()

print('Time Taken :', end_1-start_1,'seconds')
y_pred_reg = regressor.predict(X_test)
start_2 = time.time()



model = Sequential()

model.add(Dense(32,activation='relu',input_dim=1))

model.add(Dense(32,activation='relu'))

model.add(Dense(32,activation='relu'))

model.add(Dense(1))

opt  = keras.optimizers.RMSprop(learning_rate = 0.0099)

model.compile(optimizer=opt,loss='mean_squared_error')

model.fit(X_train,y_train,epochs=500)



end_2 = time.time()

print('\nTime Taken :',end_2-start_2,'seconds')
y_pred_ann = model.predict(X_test)
plt.figure(figsize=(20,10))

plt.scatter(X_train,y_train, color='red')

plt.plot(X_train,regressor.predict(X_train), color='blue') 

plt.plot(X_train,model.predict(X_train), color='green') 

plt.title('Salary vs Exp (training set)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.legend()

plt.show()
plt.figure(figsize=(20,10))

plt.scatter(X_test,y_test, color='red')

plt.plot(X_train,regressor.predict(X_train), color='blue') 

plt.plot(X_train,model.predict(X_train), color='green') 

plt.title('Salary vs Exp (test set)')

plt.xlabel('years of exp')

plt.ylabel('Salary')

plt.legend()

plt.show()
print('Criteria \t | Statistical Model \t\t| ANN ')

print("R-Sq \t\t | ", r2_score(y_test, y_pred_reg),' \t\t| ', r2_score(y_test, y_pred_ann))

print("RMSE \t\t | ", math.sqrt(mean_squared_error(y_test, y_pred_reg)),' \t\t| ', math.sqrt(mean_squared_error(y_test, y_pred_ann)))

print("Time \t\t | ", end_1 - start_1,' \t| ', end_2 - start_2)