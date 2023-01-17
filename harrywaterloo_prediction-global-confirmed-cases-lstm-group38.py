from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns; sns.set()
import missingno as msno

from sklearn.model_selection import train_test_split

# Configure visualisations
%matplotlib inline
mpl.style.use( 'ggplot' )
plt.style.use('fivethirtyeight')
sns.set(context="notebook", palette="dark", style = 'whitegrid' , color_codes=True)


# %tensorflow_version 2.x
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.python.client import device_lib

# Deep Learning Libraries
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D,AveragePooling2D
from tensorflow.keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import applications
from tensorflow.keras.regularizers import l2
tf.random.set_seed(42)
import requests
import pandas as pd
import io

BASE_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
CONFIRMED = 'time_series_covid19_confirmed_global.csv'
DEATH = 'time_series_covid19_deaths_global.csv'
RECOVERED = 'time_series_covid19_recovered_global.csv'
CONFIRMED_US = 'time_series_covid19_confirmed_US.csv'
DEATH_US = 'time_series_covid19_deaths_US.csv'

def get_covid_data(subset = 'CONFIRMED'):
    """This function returns the latest available data subset of COVID-19. 
        The returned value is in pandas DataFrame type.
    Args:
        subset (:obj:`str`, optional): Any value out of 5 subsets of 'CONFIRMED',
        'DEATH', 'RECOVERED', 'CONFIRMED_US' and 'DEATH_US' is a valid input. If the value
        is not chosen or typed wrongly, CONFIRMED subet will be returned.
    """    
    switcher =  {
                'CONFIRMED'     : BASE_URL + CONFIRMED,
                'DEATH'         : BASE_URL + DEATH,
                'RECOVERED'     : BASE_URL + RECOVERED,
                'CONFIRMED_US'  : BASE_URL + CONFIRMED_US,
                'DEATH_US'      : BASE_URL + DEATH_US,
                }

    CSV_URL = switcher.get(subset, BASE_URL + CONFIRMED)

    with requests.Session() as s:
        download        = s.get(CSV_URL)
        decoded_content = download.content.decode('utf-8')
        data            = pd.read_csv(io.StringIO(decoded_content))

    return data
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
##Online data
# death = get_covid_data(subset = 'DEATH')
# confirmed =  get_covid_data(subset = 'CONFIRMED')
# recovered = get_covid_data(subset = 'RECOVERED')
# death.to_csv(DEATH,index=False)
# confirmed.to_csv(CONFIRMED,index=False)
# recovered.to_csv(RECOVERED,index=False)

#Local data
death = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_deaths_global.csv')
confirmed = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv')
recovered = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_recovered_global.csv')
confirmed.head()

canada_cases = confirmed.loc[confirmed['Country/Region']=='Canada'].iloc[:,4:].sum(axis=0)
plot_range = range(len(canada_cases))
plt.bar(plot_range,canada_cases,label='Canada')
plt.legend()
plt.title('Total Number of COVID-19 Canada')
plt.xlabel('Day')
plt.ylabel('Number of Cases')
plt.show()
#Daily cases Canada
daily_canada = canada_cases.diff()
daily_canada = daily_canada.fillna(canada_cases[0])
plot_range = range(len(daily_canada))
plt.bar(plot_range,daily_canada,label='Canada')
plt.legend()
plt.title('Daily Number of COVID-19 Canada')
plt.show()
# print(daily_canada)
confirmed = confirmed.drop(columns=['Province/State','Lat','Long'])
death = death.drop(columns=['Province/State','Lat','Long'])
recovered = recovered.drop(columns=['Province/State','Lat','Long'])
world_cases = confirmed.iloc[:,1:].sum(axis=0)
world_death = death.iloc[:,1:].sum(axis=0)
world_recovered = recovered.iloc[:,1:].sum(axis=0)
daily_world_cases = world_cases.diff()
daily_world_cases = daily_world_cases.fillna(world_cases[0])

daily_world_death = world_death.diff()
daily_world_death = daily_world_death.fillna(world_death[0])

daily_world_recovered = world_recovered.diff()
daily_world_recovered = daily_world_recovered.fillna(world_recovered[0])


plot_range = range(len(world_cases))
plt.plot(plot_range,world_cases,color='black',label='Confirmed')
plt.bar(plot_range,world_cases,color='black')
plt.plot(plot_range,world_death,color='red',label='Death')
plt.bar(plot_range,world_death,color='red')
plt.plot(plot_range,world_recovered,color='green',label='Recovered')
plt.bar(plot_range,world_recovered,color='green')
plt.legend()
plt.xlabel('Time', size=10)
plt.ylabel('# Cases', size=10)
plt.xticks(size=10)
plt.title('Total cases - Word')
plt.show()


plot_range = range(len(daily_world_cases))
plt.plot(plot_range,daily_world_cases,color='black',label='Confirmed')
plt.bar(plot_range,daily_world_cases,color='black')
plt.plot(plot_range,daily_world_death,color='red',label='Death')
plt.bar(plot_range,daily_world_death,color='red')
plt.plot(plot_range,daily_world_recovered,color='green',label='Recovered')
plt.bar(plot_range,daily_world_recovered,color='green')
plt.legend()
plt.xlabel('Time', size=10)
plt.ylabel('# Cases', size=10)
plt.xticks(size=10)
plt.title('Daily new cases - Word')
plt.show()

print(daily_world_cases.isnull().any())
print(daily_world_cases.isna().any())
print(daily_world_death.isnull().any())
print(daily_world_death.isna().any())
print(daily_world_recovered.isnull().any())
print(daily_world_recovered.isna().any())
data = pd.DataFrame([])
# data['Date'] = world_cases.index.tolist()
data['world_confirmed'] = world_cases.values
data['daily_world_confirmed'] = daily_world_cases.values
data['world_death'] = world_death.values
data['daily_world_death'] = daily_world_death.values
data['world_recovered'] = world_recovered.values
data['daily_world_recovered'] = daily_world_recovered.values
print("Yesterday")
data.tail(1)

features_list = ['world_death','world_recovered','world_confirmed']
features = data[features_list]
labels = data['world_confirmed']


def create_data(dataset, target, start_index, end_index, history_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset)

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    data.append(dataset[indices])
    labels.append(target[i])


  return np.array(data), np.array(labels)
from sklearn.preprocessing import MinMaxScaler
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaler_labels = MinMaxScaler(feature_range=(0, 1))

future = 14 #Number of predictions to the future (days)

#How much data for training
TRAIN_SPLIT = len(data) - future -1 #How much data for training
print(TRAIN_SPLIT)

#Data scaling
dataset = features.values
scaler_features.fit(dataset[:TRAIN_SPLIT])
scaler_labels.fit(dataset[:TRAIN_SPLIT,2].reshape(-1,1))
dataset = scaler_features.transform(dataset)
print(dataset.shape)
# print(dataset)
past_history = 5

x_train, y_train = create_data(dataset, dataset[:, 2], 0,
                              TRAIN_SPLIT, past_history)
x_val, y_val = create_data(dataset, dataset[:, 2],
                            TRAIN_SPLIT, None, past_history)

print(y_train.shape)
print(x_train.shape)
print(y_val.shape)
print(x_val.shape)
print ('Single window of past history')
print (x_train[1])
print ('\n Target value to predict: ')
print (y_train[1])
lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, input_shape=x_train.shape[-2:],activation='relu'),
    tf.keras.layers.Dense(1)
])
# reduce_lr = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999 )
lstm_model.compile(optimizer='adam', loss='mae')


history = lstm_model.fit(x_train,y_train, batch_size =8,epochs = 100, validation_data=(x_val,y_val), verbose=2)
for sample in range(4):
  print('Predict')
  predict = lstm_model.predict(x_val)
  # print(predict[sample])
  print(scaler_labels.inverse_transform(predict[sample].reshape(1,-1)))
  print("Actual")
  # print(y_val[sample])
  print(scaler_labels.inverse_transform(y_val[sample].reshape(1,-1)))
  # show_plot([x_val_single[sample,2], y_val_single[sample],predict[sample]], 0, 'Sample Example')
  # plt.show()
predict_confirmed = np.copy(dataset)
future = 14
for i in range(future):
#     print(TRAIN_SPLIT+i+1)
    sample = predict_confirmed[TRAIN_SPLIT-past_history+i:TRAIN_SPLIT+i]
    # print(sample.shape)
    sample = sample.reshape((1,sample.shape[0],sample.shape[1]))
    predict = lstm_model.predict(sample)
#     print("Predict: ", predict)
#     print("Actual: ", dataset[TRAIN_SPLIT+i+1,2])
    try:
        predict_confirmed[TRAIN_SPLIT+i+1,2] = predict
    except:
        print()
    
# i = 0
# sample = dataset[TRAIN_SPLIT-5+i:TRAIN_SPLIT+i]
# print(sample)
# sample = sample.reshape((1,5,3))
# predict = simple_lstm_model.predict(sample)
# print("Predict: ", predict)
# print(dataset[TRAIN_SPLIT+i])
# print("Actual: ",dataset[TRAIN_SPLIT+i,2])
# dataset_new[TRAIN_SPLIT+i+1,2] = predict
# print(dataset_new[TRAIN_SPLIT+i+1,2])
# print(dataset_new[TRAIN_SPLIT-5+i+1:TRAIN_SPLIT+i+1])
# plt.plot(dataset_new[:,2])
# plt.plot(dataset[:,2])
# dataset_scale = scaler_features.inverse_transform(dataset)
predict_confirmed_scale = scaler_features.inverse_transform(predict_confirmed)

plt.figure(figsize=(10,8))
plt.plot(features["world_confirmed"],label="Actual",color='g')
plt.plot(predict_confirmed_scale[:,2],label="Prediction",marker="o")
plt.xlabel("Day n-th since 1/22/2020", size = 20)
plt.ylabel("Number of cases", size = 20)
plt.title("World - number of confirmed cases", size = '20')
plt.legend()
plt.show()

predict_daily = pd.DataFrame(predict_confirmed_scale[:,2]).diff()
predict_daily.fillna(0)
# print(predict_daily.shape)
# print(data['daily_world_confirmed'].shape)
plt.figure(figsize=(10,8))
plt.plot(data['daily_world_confirmed'],label="Actual",color='g')
plt.plot(predict_daily,label="Prediction",marker="o")
plt.xlabel("Day n-th since 1/22/2020", size = 20)
plt.ylabel("Number of cases", size = 20)
plt.title("World - number of confirmed cases", size = '20')
plt.legend()
plt.show()
