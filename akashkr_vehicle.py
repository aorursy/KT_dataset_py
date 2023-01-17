from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras.layers as L
import matplotlib.pyplot as plt
import plotly.express as px
import tensorflow as tf
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import os

data_dir = '../input/vehicle-data/'
# Reading Data
vehicle = pd.read_csv(os.path.join(data_dir, 'Train_Vehicletravellingdata.csv'))
weather = pd.read_csv(os.path.join(data_dir, 'Train_WeatherData.csv'))
train = pd.read_csv(os.path.join(data_dir, 'Train.csv'))
# Concat vehicle and weather data
cons_df = pd.concat([vehicle, weather.drop(columns=['ID', 'Date time'])], axis=1)
# Merge with train data
cons_df = cons_df.merge(train, on=['ID'])
# Sample Data
cons_df.head()
# About columns
cons_df.info()
# Let's see the count of target variable
# driving_style_df = cons_df['DrivingStyle'].value_counts().to_frame()
# driving_style_df.columns = ['count']
# driving_style_df['Labels'] = driving_style_df.index

# fig = px.bar(driving_style_df, x='Labels', y='count')
# fig.show()
# Count of missing values in columns
# missing_values = cons_df.isnull().sum()
# missing_values = missing_values[missing_values > 0].sort_values()

# missing_values = missing_values.to_frame()
# missing_values.columns = ['count']
# missing_values.index.names = ['Name']
# missing_values['Name'] = missing_values.index

# fig = px.bar(missing_values, x='Name', y='count')
# fig.show()
# Checking distribution of Speed
# df = px.data.tips()
# fig = px.box(cons_df, y="Speed of the vehicle (kph)", points="all")
# fig.show()
cons_df['Road Condition'].value_counts()
# Removing outliers
cons_df = cons_df.loc[
    (cons_df['Speed of the vehicle (kph)']> 20) & 
    (cons_df['Speed of the vehicle (kph)']<140)
]
label_encoding_columns = [
    'Road Condition', 'Precipitation', 'Precipitation intensity', 'Day time'
]
cons_df[label_encoding_columns]
cons_df[label_encoding_columns] = cons_df[label_encoding_columns].apply(LabelEncoder().fit_transform)
train_df = cons_df.drop(columns=['ID of the preceding vehicle','Date time', 'DrivingStyle'])
missing_values_columns = list(missing_values['Name'])

for col in missing_values_columns:
    train_df[col] = train_df[col].fillna(train_df[col].mean())
train_df.head()
# Let's make some features
cons_df['speed_ratio'] = cons_df['Speed of the vehicle (kph)']/cons_df['Speed of the preceding vehicle']
cons_df['time_ratio1'] = cons_df['Time gap with the preceeding vehicle in seconds']/cons_df['Speed of the vehicle (kph)']
cons_df['time_ratio2'] = cons_df['Time gap with the preceeding vehicle in seconds']/cons_df['Speed of the preceding vehicle']
scaling_columns = [
    'Speed of the vehicle (kph)', 'Speed of the preceding vehicle', 'Length of preceding vehicle',
    'Time gap with the preceeding vehicle in seconds', 'Weather details-Air temperature', 'Weight of the preceding vehicle',
    'Relative humidity', 'Wind direction', 'Wind speed in m/s', 'Length of vehicle in cm', 'weight of vehicle in kg',
    'Number of axles', 'Number of axles'
]
scaler = StandardScaler()
train_df[scaling_columns] = scaler.fit_transform(train_df[scaling_columns])
train_df
sequences = list()

for name, group in tqdm(train_df.groupby(['ID'])):
    sequences.append(group.drop(columns=['ID']).values)
    
# train_values = np.asarray(train_values)
len_sequences = []
for one_seq in sequences:
    len_sequences.append(len(one_seq))
pd.Series(len_sequences).describe()
#Padding the sequence with the values in last row to max length
to_pad = 112
new_seq = []
for one_seq in sequences:
    len_one_seq = len(one_seq)
    last_val = one_seq[-1]
    n = to_pad - len_one_seq
   
    to_concat = np.repeat(one_seq[-1], n).reshape(17, n).transpose()
    new_one_seq = np.concatenate([one_seq, to_concat])
    new_seq.append(new_one_seq)
final_seq = np.stack(new_seq)

#truncate the sequence to length 60
# from tf.keras.preprocessing import sequence
seq_len = 60
final_seq=tf.keras.preprocessing.sequence.pad_sequences(final_seq, maxlen=seq_len, padding='post', dtype='float', truncating='post')
# from keras.utils.np.utils import to_categorical
# y_train = to_categorical(y_train)

target = pd.get_dummies(train['DrivingStyle'])
target = np.asarray(target)
X_train, X_test, y_train, y_test = train_test_split(final_seq, target, test_size=0.20, random_state=34)
model = tf.keras.models.Sequential()
model.add(L.LSTM(128, dropout=0.2, input_shape=(seq_len, 17), return_sequences=True))
model.add(L.LSTM(64, dropout=0.2, input_shape=(seq_len, 17), return_sequences=True))
model.add(L.LSTM(64, dropout=0.2))
model.add(L.Dense(3, activation='softmax'))
# adam = tf.optimizers.Adam(lr=0.1, clipvalue=0.5)
adam = tf.keras.optimizers.Adam(lr=0.001)
# sgd = tf.keras.optimizers.SGD(lr=1)
sgd = tf.keras.optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)

# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=84,
    validation_data=(X_test, y_test),
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(patience=5),
    ]
)

