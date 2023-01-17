import os
import joblib
import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
INPUT_POINTS = '../input/soybean_2019_2020_samples.gpkg'
INPUT_DATA = '../input/soybean_southamerica_mod13q1_evi_2000_2019.csv'

FIELD_ID = 'fid'
FIELD_TIMESTAMP = 'timestamp'


TRAIN_FEATURES = ['evi']
TRAIN_VALUES = 500

PREDICT_FEATURE = 'evi'
PREDICT_START_DATE = '2019-01-01'
PREDICT_END_DATE = '2019-12-31'
PREDICT_VALUES = 23

SCALER_PATH = '/kaggle/working/lstm_scaler.save'
MODEL_PATH = '/kaggle/working/lstm_trained_model.h5'
gdf = gpd.read_file(INPUT_POINTS)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# We restrict to South America.
ax = world[world.continent == 'South America'].plot(color='white', edgecolor='black', figsize=(20, 15))

# We can now plot our ``GeoDataFrame``.
gdf.plot(ax=ax, color='orange')
plt.title('Soybean South America: 50k samples', fontsize=16)
plt.show()
df = pd.read_csv(INPUT_DATA)
df.tail()
all_series = df[FIELD_ID].dropna().unique()
len(all_series)
trainining_ids, test_ids = train_test_split(all_series, test_size=0.10, random_state=101)
len(trainining_ids), len(test_ids)
def load_dataset(ids, dataset, train_values=500, predict_values=23):
    x = []
    y = []

    for id in tqdm.tqdm(ids):
        data = dataset[(dataset[FIELD_ID] == id)]
        index = pd.DatetimeIndex(pd.to_datetime(data[FIELD_TIMESTAMP],  unit='ms'))
        data.index = index

        x_data = data.loc[:PREDICT_START_DATE][TRAIN_FEATURES].dropna()
        y_data = data.loc[PREDICT_START_DATE:PREDICT_END_DATE][PREDICT_FEATURE].dropna()

        if len(y_data) != predict_values:
            continue

        x_data = tf.keras.preprocessing.sequence.pad_sequences([x_data], maxlen=train_values, dtype='float32')[0]

        x.append(x_data)
        y.append(y_data.values)

    return np.array(x), np.array(y)

train_x_data, train_y_data  = load_dataset(ids=trainining_ids, 
                             dataset=df, 
                             train_values=TRAIN_VALUES, 
                             predict_values=PREDICT_VALUES)

train_x_data.shape, train_y_data.shape
lstm_model = Sequential()

input_shape = (train_x_data.shape[1], 1)

lstm_model.add(LSTM(128, return_sequences=True, input_shape=input_shape))

lstm_model.add(LSTM(128))

lstm_model.add(Dense(train_y_data.shape[1]))

lstm_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

tf.keras.utils.plot_model(lstm_model, show_shapes=True, to_file='/kaggle/working/model.png')
if os.path.exists(SCALER_PATH):
    print("Loading saved scaler...")
    scaler = joblib.load(SCALER_PATH) 
else: 
    scaler = MinMaxScaler()
    scaler.fit(train_x_data.reshape(-1, train_x_data.shape[-1]))
    joblib.dump(scaler, SCALER_PATH)
for i, value in enumerate(train_x_data):
    value = value[~np.isnan(value).any(axis=1)]
    train_x_data[i] = scaler.transform(value)

for i, value in enumerate(train_y_data):
    value = value[~np.isnan(value).any()]
    train_y_data[i] = scaler.transform(value)
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True
)

if os.path.exists(MODEL_PATH):
    print("Loading trained model...")
    lstm_model = tf.keras.models.load_model(MODEL_PATH)
else:
    history = lstm_model.fit(train_x_data, train_y_data, 
                        epochs=100, 
                        batch_size=32, 
                        validation_split=0.25, 
                        verbose=1, 
                        callbacks=[callback]) 

    # plot history
    plt.figure(figsize=(20,8))
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.ylabel('Model loss')
    plt.xlabel('Epochs')
    plt.show()
    plt.savefig('/kaggle/working/model_loss.png', dpi=300, bbox_inches='tight')
test_x_data, test_y_data  = load_dataset(ids=test_ids, 
                             dataset=df, 
                             train_values=TRAIN_VALUES, 
                             predict_values=PREDICT_VALUES)

test_x_data.shape, test_y_data.shape

for i, value in enumerate(test_x_data):
    value = value[~np.isnan(value).any(axis=1)]
    test_x_data[i] = scaler.transform(value)

for i, value in enumerate(test_y_data):
    value = value[~np.isnan(value).any()]
    test_y_data[i] = scaler.transform(value)
lstm_model.evaluate(x=test_x_data, y=test_y_data)
def expected_vs_predicted(input, expected, predicted):
    expected_values = expected.copy()
    
    index = expected.index.to_pydatetime()

    expected_values.index = index
    
    fitted_series = pd.Series(predicted)
    fitted_series.index=index

    plt.figure(figsize=(20,15))
    plt.subplot(211)

    # plotar gráfico
    plt.plot(input, color='green')
    plt.plot(fitted_series, linestyle='--', color='orange')
    plt.ylim(0, 1)
    plt.legend(['real', 'previsto'], loc='upper left')

    rmse = np.sqrt(mean_squared_error(expected_values, predicted))

    return rmse
for id in test_ids[:50]:
    data = df[(df[FIELD_ID] == id)]
    index = pd.DatetimeIndex(pd.to_datetime(data[FIELD_TIMESTAMP],  unit='ms'))
    data.index = index

    input = data.loc[:PREDICT_START_DATE][TRAIN_FEATURES].dropna()
    input = tf.keras.preprocessing.sequence.pad_sequences([input], maxlen=TRAIN_VALUES, dtype='float32')[0]

    expected = data.loc[PREDICT_START_DATE:PREDICT_END_DATE][PREDICT_FEATURE]
    expected = expected.dropna()

    if len(expected) != PREDICT_VALUES:
        continue
        
    predicted = lstm_model.predict(np.array([input]))[0]

    # inverse transform
    reescaled_expected = scaler.inverse_transform(expected.values.reshape(expected.values.shape[0], 1)).flatten()
    reescaled_predicted = scaler.inverse_transform(predicted.reshape(predicted.shape[0], 1)).flatten()

    reescaled_expected = pd.Series(reescaled_expected, index=expected.index)

    rmse = expected_vs_predicted(data[TRAIN_FEATURES].loc['2015-01-01':].dropna(), reescaled_expected, reescaled_predicted)
    plt.title("Actual vs Predicted Vegetation Index | Sample ID: %s. RMSE: %.3f" % (str(id), rmse, ), fontsize=20)
    plt.ylabel('Vegetation Index (EVI)', fontsize=16)
    plt.xlabel(None)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(['Actual', 'Predicted'], fontsize=16)
    plt.show()