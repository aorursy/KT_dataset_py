import pandas as pd
import numpy as np

BASE_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
CONFIRMED = 'time_series_covid19_confirmed_global.csv'
URL = BASE_URL + CONFIRMED

data_download = pd.read_csv(URL)
data_download = data_download[data_download['Country/Region'] == 'United Kingdom']
province = data_download['Province/State'].values
province = ['Other' if item is np.nan else item for item in province]
data_download = data_download.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long']).values

# Store all the data into store based on province. Finally, add them up to form a total.
store = {}
store_scaler = {}
store_model = {}

for index,prov in enumerate(province):
    store[prov] = np.reshape(data_download[index], (-1, 1))
store['Total'] = np.reshape(np.sum(data_download, axis=0), (-1, 1))

print(province)
from sklearn.preprocessing import MinMaxScaler

def _data_norm(data_train, data_test):
    # This scaler will also be used for future data recovery
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_train_norm = scaler.fit_transform(data_train)
    data_test_norm = scaler.transform(data_test)
    return data_train_norm, data_test_norm, scaler
from sklearn.model_selection import train_test_split

# Split data
def _data_split(data, test_ratio):
    data_train, data_test = train_test_split(data, test_size=test_ratio, shuffle=False)
    return np.reshape(data_train, (-1,1)), np.reshape(data_test, (-1,1))
# Generate data based on training_step and prediction_step
def _data_generator(data, training_step, prediction_step):
    data_X = []
    data_y = []
    for i in range(len(data) - training_step - prediction_step):
        data_X.append(data[i:i+training_step])
        data_y.append(data[i + training_step:i + training_step + prediction_step])
    return np.array(data_X), np.array(data_y)
# Generate data by the above methods
def data_prepare(region, test_ratio, training_step, prediction_step):
    if region not in store:
        raise ValueError('data_prepare():' + str(region) + ' is not a supported region!')
    data = store[region]
    train, test         = _data_split(data, test_ratio)
    train, test, scaler = _data_norm(train, test)
    X_train, y_train    = _data_generator(train, training_step, prediction_step)
    X_test,  y_test     = _data_generator(test, training_step, prediction_step)
    X_train = np.reshape(X_train, (np.shape(X_train)[0], -1, 1))
    X_test  = np.reshape(X_test, (np.shape(X_test)[0], -1, 1))
    # Store the current scaler for the region for later data recovery
    store_scaler[region] = scaler
    return X_train, y_train, X_test, y_test
        
import tensorflow as tf
from tensorflow import keras

def GRU(training_step, prediction_step):
    model = keras.Sequential([
        keras.layers.GRU(units=256, input_shape=(training_step,1), dropout=0.2, return_sequences=True),
        keras.layers.GRU(units=128, dropout=0.2, return_sequences=False),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dense(units=prediction_step)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
test_ratio      = 0.15
training_step   = 5
# Notice, if prediction_step is not 1, other functions might crash
prediction_step = 1

# Start training models for all regions
epochs = 200

for region in store: 
    print('Start training (' + str(epochs) + ' epochs): ' + region)
    X_train, y_train, X_test, y_test = data_prepare(region=region, test_ratio=test_ratio, training_step=training_step, prediction_step=prediction_step)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    model = GRU(training_step, prediction_step)
    model.fit(X_train,y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)
    store_model[region] = model
    print('Evaulate on ' + region + ' :' + str(model.evaluate(X_test, y_test)) + '\n')
# Used to predict the future data
def data_predict_recover(region, prediction_days):
    model  = store_model[region]
    scaler = store_scaler[region]
    data   = scaler.transform(store[region])[-training_step:].tolist()
    for day in range(prediction_days):
        pred = model.predict(np.reshape(data[-training_step:], (1, -1, 1)))
        data.append([float(pred)])
        if len(data) > prediction_days:
            data = data[1:]
    return scaler.inverse_transform(data)
# Doing data predictions
prediction_days = 10
store_predictions = {}

for region in store:
    store_predictions[region] = data_predict_recover(region=region, prediction_days=prediction_days)

print(store_predictions)
import matplotlib.pyplot as plt

print('Prediction days ' + str(prediction_days))

plt.subplots(3,4,figsize=(25,20))
for i, region in enumerate(store.keys()):
    data = np.reshape(store[region],(1,-1)).tolist()[0]
    data_prediction = np.reshape(store_predictions[region], (1,-1)).tolist()[0]
    plt.subplot(3,4,i+1)
    plt.plot(list(range(len(data))),data, '-.', label='Current')
    plt.plot(list(range(len(data), len(data) + prediction_days)),data_prediction, '.', label='Prediction')
    plt.xlabel('Day')
    plt.ylabel('Confirmed Cases')
    plt.title('UK ' + region + ' COVID-19 Confirmed Cases' )
    plt.legend()
plt.show()
import pandas as pd
import numpy as np

BASE_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
DEATH = 'time_series_covid19_deaths_global.csv'
URL = BASE_URL + DEATH

data_download = pd.read_csv(URL)
data_download = data_download[data_download['Country/Region'] == 'United Kingdom']
province = data_download['Province/State'].values
province = ['Other' if item is np.nan else item for item in province]
data_download = data_download.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long']).values

# Store all the data into store based on province. Finally, add them up to form a total.
store = {}
store_scaler = {}
store_model = {}

for index,prov in enumerate(province):
    store[prov] = np.reshape(data_download[index], (-1, 1))
store['Total'] = np.reshape(np.sum(data_download, axis=0), (-1, 1))

print(province)
import matplotlib.pyplot as plt

print('Prediction days ' + str(prediction_days))

plt.subplots(3,4,figsize=(25,20))
for i, region in enumerate(store.keys()):
    data = np.reshape(store[region],(1,-1)).tolist()[0]
    data_prediction = np.reshape(store_predictions[region], (1,-1)).tolist()[0]
    plt.subplot(3,4,i+1)
    plt.plot(list(range(len(data))),data, '-.', label='Current')
    plt.plot(list(range(len(data), len(data) + prediction_days)),data_prediction, '.', label='Prediction')
    plt.xlabel('Day')
    plt.ylabel('Death Cases')
    plt.title('UK ' + region + ' COVID-19 Death Cases' )
    plt.legend()
plt.show()