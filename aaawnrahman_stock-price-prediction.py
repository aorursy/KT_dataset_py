# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#reference_help : https://stackabuse.com/time-series-analysis-with-lstm-using-pythons-keras-library

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv("/kaggle/input/stock-price/AAPL.csv")
df.head()
df.tail()
df.plot(x='Date', y='Open')
df.plot(x='Date', y='Close')
df.info()
training_data = df.iloc[0:1260,1:2].values

training_data
#data normalization......

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0, 1))

training_data = scaler.fit_transform(training_data)
#Convert Training Data to Right Shape

features_set = []

labels = []

for i in range(60,1260) :

    features_set.append(training_data[i-60:i,0])

    labels.append(training_data[i,0])

    

#convert feature_set and labels in numpy array .....

features_set , labels = np.array(features_set) , np.array(labels)



#convert the feature_set into a 3 dimentional format.....

features_set = np.reshape(features_set, (features_set.shape[0] , features_set.shape[1] , 1))



    

#features_set shape......

print(features_set.shape)

print(labels.shape)

#import libraries.......

from keras.models import Sequential

from keras.layers import Dense, LSTM, Dropout
#build a LSTM model......

model = Sequential()



model.add(LSTM(units= 50, return_sequences= True , input_shape = (features_set.shape[1], 1) ))

model.add(Dropout(0.2))

model.add(LSTM(units = 50 , return_sequences = True))

model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))

model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = False))

model.add(Dropout(0.2))

model.add(Dense(units= 1))
model.summary()
#compiling the model.....

model.compile(optimizer = 'adam' , loss = 'mean_squared_error', metrics=['accuracy'])

print(features_set.shape)

print(labels.shape)
fitted_model =  model.fit(features_set , labels , epochs = 100 , batch_size = 32)
import pandas as pd

AAPL_test = pd.read_csv("/kaggle/input/test-data/AAPL_test.csv")

AAPL_test.info()

AAPL_test.head()
testing_dataset = AAPL_test.iloc[:,1:2].values

x = pd.DataFrame(testing_dataset)

y = pd.DataFrame(training_data)

total = pd.concat([df['Open'] ,AAPL_test['Open']] , axis = 0)

print(type(total))

total = total.drop_duplicates()


total.shape
#prepare the test input.......

test_inputs = total[len(total) - len(AAPL_test) - 60:].values
#reshape the test data........

test_inputs = test_inputs.reshape(-1,1)

test_inputs = scaler.transform(test_inputs)

print(testing_dataset.shape)

print(test_inputs.shape)

test_features = []

for i in range(60, 298):

    test_features.append(test_inputs[i-60:i, 0])
test_features = np.array(test_features)

test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
#prediction

predictions = model.predict(test_features)
#inverse scale

predictions = scaler.inverse_transform(predictions)
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

plt.plot(testing_dataset, color='blue', label='Actual Apple Stock Price')

plt.plot(predictions , color='red', label='Predicted Apple Stock Price')

plt.title('Apple Stock Price Prediction')

plt.xlabel('Date')

plt.ylabel('Apple Stock Price')

plt.legend()

plt.show()