import pandas 

from keras.models import Sequential

from keras.layers import Dense



import numpy

numpy.random.seed(7)



data= pandas.read_csv('../input/Road Accidents-Regression.csv')



data.head()
data = data.values

X = data[:,0:5]

Y = data[:,5]
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1

                                                    ,random_state = 0)



X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.11

                                                   ,random_state = 0)

X_train.shape
X_val.shape
X_test.shape
model = Sequential()



model.add(Dense(11, input_dim=7, kernel_initializer='uniform', activation='relu'))



model.add(Dense(5, kernel_initializer='uniform', activation='relu'))



model.add(Dense(3, kernel_initializer='uniform', activation='relu'))



model.add(Dense(2, kernel_initializer='uniform'))



model.summary()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])