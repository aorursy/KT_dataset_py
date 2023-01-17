%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
!ls
data_path = '../input/hour.csv'



rides = pd.read_csv(data_path)
rides.head()
rides[:24*10].plot(x='dteday', y='cnt')
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']

for each in dummy_fields:

    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)

    rides = pd.concat([rides, dummies], axis=1)



fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 

                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']

data = rides.drop(fields_to_drop, axis=1)

data.head()
quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']

# Store scalings in a dictionary so we can convert back later

scaled_features = {}

for each in quant_features:

    mean, std = data[each].mean(), data[each].std()

    scaled_features[each] = [mean, std]

    data.loc[:, each] = (data[each] - mean)/std
# Save data for approximately the last 21 days 

test_data = data[-21*24:]



# Now remove the test data from the data set 

data = data[:-21*24]



# Separate the data into features and targets

target_fields = ['cnt', 'casual', 'registered']

features, targets = data.drop(target_fields, axis=1), data[target_fields]

test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
# Hold out the last 60 days or so of the remaining data as a validation set

train_features, train_targets = features[:-60*24], targets[:-60*24]

val_features, val_targets = features[-60*24:], targets[-60*24:]
train_targets.shape
train_features.shape
from keras.models import Sequential

from keras.layers import Dense



model = Sequential()

model.add(Dense(128, input_dim=56, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(32, activation='relu'))

model.add(Dense(16, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(3, activation='softmax'))



model.summary()
# compile the keras model

model.compile(loss='mse', optimizer='adam', metrics=['accuracy','mse'])

# fit the keras model on the dataset

model.fit(train_features, train_targets, validation_data=(val_features, val_targets), epochs=30, batch_size=10)



fig, ax = plt.subplots(figsize=(8,4))



mean, std = scaled_features['cnt']

predictions = network.run(test_features).T*std + mean

ax.plot(predictions[0], label='Prediction')

ax.plot((test_targets['cnt']*std + mean).values, label='Data')

ax.set_xlim(right=len(predictions))

ax.legend()



dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])

dates = dates.apply(lambda d: d.strftime('%b %d'))

ax.set_xticks(np.arange(len(dates))[12::24])

_ = ax.set_xticklabels(dates[12::24], rotation=45)