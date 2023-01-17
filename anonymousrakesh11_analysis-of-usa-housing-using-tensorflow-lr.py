# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#          || **  Ignore my English grammar and spelling mistske ** || (-_-)



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))

import matplotlib.pyplot as plt

import seaborn as sns

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam



# Any results you write to the current directory are saved as output.
data= pd.read_csv("../input/USA_Housing.csv")

data.head()
data.info()
print('check if any null value:- \n', (data.isnull().sum()))

sns.pairplot(data)
plt.figure(figsize=(14,10))

sns.heatmap(data.corr(),annot=True)
# address is not seem to useful so we can delete this column:-

# we devide the data into dependent veriable and independent veriable(where X is independent veriable and y is dependent veriable)



X= data.drop(['Address','Price'], axis=1)

y= data['Price']
#using Standard Scaler we sacle the independent veriables data:-



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

X= scaler.transform(X)

print(X)
#splitting the data into trainning and testing sets:-



from sklearn.model_selection import train_test_split

X_train,  X_test,y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#Apply linear regression to our model :- 



from sklearn.linear_model import LinearRegression 

regressor = LinearRegression()

regressor.fit(X_train, y_train)
#Now we can see  the output of 1st row of our data and compare actual price value  to the predicted price value 



y_predict= regressor.predict(X_test)

print('Predicted Value :',y_predict[0])

print('Actual Value :',y_test.values[0])

test = pd.DataFrame({'Predicted':y_predict,'Actual':y_test})

test = test.reset_index()

test = test.drop(['index'],axis=1)
# Graph is output of first 50 test datasets  and its predicted value :-





plt.figure(figsize=(20,12))

plt.plot(test[:50])

plt.legend(['Actual','Predicted'])

sns.jointplot(x='Actual',y='Predicted',data=test,kind='reg')
from sklearn.metrics import r2_score, mean_squared_error



score = r2_score(regressor.predict(X_train), y_train)

error = mean_squared_error(regressor.predict(X_train), y_train)
print(score,error)
regressor.coef_
Z= data.drop(['Address'], axis=1)

Z
pd.DataFrame(regressor.coef_, index=Z.columns[:-1], columns=['Values'])
score_test = r2_score(regressor.predict(X_test),y_test)

print(score_test)
LR_mse = mean_squared_error(y_test, y_predict)

LR_rmse = np.sqrt(LR_mse)

LR_rmse
DNN_regressor = Sequential()

DNN_regressor.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu', input_dim = 5))

DNN_regressor.add(Dense(output_dim = 64, init = 'uniform', activation='relu'))

DNN_regressor.add(Dense(output_dim = 32, init = 'uniform', activation='relu'))

DNN_regressor.add(Dense(output_dim = 16, init = 'uniform', activation='relu'))

DNN_regressor.add(Dense(output_dim = 8, init = 'uniform', activation='relu'))

DNN_regressor.add(Dense(output_dim = 4, init = 'uniform', activation='relu'))

DNN_regressor.add(Dense(output_dim = 1, init = 'uniform', activation='linear'))



DNN_regressor.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

DNN_regressor.fit(X_train, y_train, epochs=100, batch_size=10)
y_pred_DNN = DNN_regressor.predict(X_test)
DNN_mse = mean_squared_error(y_test, y_pred_DNN)

DNN_rmse = np.sqrt(DNN_mse)

DNN_rmse