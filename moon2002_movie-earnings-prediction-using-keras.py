import pandas as pd

import numpy as np
train = pd.read_csv('../input/sales-data/sales_data_training.csv')

train.head()
test = pd.read_csv('../input/sales-data/sales_data_test.csv')

test.head()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))

scaled_train = scaler.fit_transform(train)

scaled_test = scaler.transform(test) #calling transform ensures the test dataset is scaled by the same amount as the test dataset
scaled_train = pd.DataFrame(scaled_train, columns=train.columns.values)

scaled_test = pd.DataFrame(scaled_test, columns=test.columns.values)
#values for rescaling after making predictions

print('total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}'.format(scaler.scale_[8], scaler.min_[8]))
from keras.models import Sequential

from keras.layers import *
X = scaled_train.drop('total_earnings', axis=1).values

y = scaled_train[['total_earnings']].values



X_test = scaled_test.drop('total_earnings', axis=1).values

y_test = scaled_test[['total_earnings']].values
#define model, (requires trial and error to figure out how dense the layers should be)

model = Sequential()

model.add(Dense(50, input_dim=9, activation='relu')) #50 layers, 9 because there are 9 characteristics for each game, relu activation function allows you to model more complex non-linear functions

model.add(Dense(100, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(1, activation='linear')) #linear activation is default so don't actually need to put it

model.compile(loss='mean_squared_error', optimizer='adam')
#train model

model.fit(

    X,

    y,

    epochs=50,

    shuffle=True,

    verbose=2

)
test_error_rate = model.evaluate(X_test, y_test, verbose=0)

print('MSE for test data is:{}'.format(test_error_rate)) 

#the smaller the MSE the better, means predictions are very close to expected values
#make a prediction on prescaled new data

proposed_new_product = pd.read_csv("../input/proposed-new-products/proposed_new_product.csv").values



prediction = model.predict(proposed_new_product)

prediction = prediction [0][0]



#reverse scaling

prediction = (prediction + 0.115913) / 0.0000036968

prediction
#save model to disk

model.save('trained_model.h5')