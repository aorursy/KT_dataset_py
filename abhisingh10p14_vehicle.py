from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras.layers as L

import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

data_dir = '../input/vehicle-data/'
vehicle = pd.read_csv(os.path.join(data_dir, 'Train_Vehicletravellingdata.csv'))
weather = pd.read_csv(os.path.join(data_dir, 'Train_WeatherData.csv'))
train = pd.read_csv(os.path.join(data_dir, 'Train.csv'))
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"/

lane = vehicle['Lane of the road']
lane.hist()
plt.show()
#column names of veichles :
names = list(vehicle.columns.values)
print(names)

# vehicle.head()
roadCondition = vehicle['Road Condition']
roadCondition.hist()
plt.show()
speedOfvehicle = vehicle['Speed of the vehicle (kph)']
speedOfvehicle.hist()
plt.show()
namesWithoutId = [  'Lane of the road', 'Speed of the vehicle (kph)', 
                  'Speed of the preceding vehicle', 'Weight of the preceding vehicle', 'Length of preceding vehicle', 
                  'Time gap with the preceeding vehicle in seconds', 'Road Condition']
#correlation Matrix
data =  vehicle[namesWithoutId]
correlations = data.corr()
# plot correlation matrix
fig = plt.figure(figsize=(50,50))
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(namesWithoutId)
ax.set_yticklabels(namesWithoutId)
plt.show()
#boxplot for outliers for speedOfvehicle

#data = data.drop(data[(data['Speed of the vehicle (kph)'] >140) | (data['Speed of the vehicle (kph)']< 20)].index)

fig1 = plt.figure(figsize=(25,25))
plt.boxplot(data['Speed of the vehicle (kph)'])
plt.show()


train['DrivingStyle'] = train['DrivingStyle']
train.head()
#correlation Matrix for vehihcle data 

colNames =  ['Length of vehicle in cm', 'weight of vehicle in kg', 'Number of axles', 'DrivingStyle'] 
dataVehicle =  train[colNames]
correlations = dataVehicle.corr()
# plot correlation matrix
fig = plt.figure(figsize=(25,25))
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(colNames)
ax.set_yticklabels(colNames)
plt.show()
# #outliers for weight of vehicle in kg
fig1 = plt.figure(figsize=(25,25))
plt.boxplot(dataVehicle['weight of vehicle in kg'])
plt.show()

# #outliers for Length of vehicle in cm
fig1 = plt.figure(figsize=(25,25))
plt.boxplot(dataVehicle['Length of vehicle in cm'])
plt.show()

cons_df = pd.concat([vehicle, weather.drop(columns=['ID', 'Date time'])], axis=1)
cons_df = cons_df.merge(train, on=['ID'])
weather.head()
label_encoding_columns = ['Road Condition', 'Precipitation', 'Precipitation intensity', 'Day time']
cons_df[label_encoding_columns] = cons_df[label_encoding_columns].apply(LabelEncoder().fit_transform)
train_df = cons_df.drop(columns=['ID of the preceding vehicle', 'Date time', 'DrivingStyle'])
train_df = train_df.fillna(0)
train_df.describe()
train_df.info()
#removing outliers 
#1. speed of vehicles :
train_df = train_df.drop(train_df[(train_df['Speed of the vehicle (kph)'] >140) | (train_df['Speed of the vehicle (kph)']< 20)].index)

scaling_columns = [
    'Speed of the vehicle (kph)', 'Speed of the preceding vehicle', 'Length of preceding vehicle',
    'Time gap with the preceeding vehicle in seconds', 'Weather details-Air temperature', 'Weight of the preceding vehicle',
    'Relative humidity', 'Wind direction', 'Wind speed in m/s', 'Length of vehicle in cm', 'weight of vehicle in kg',
    'Number of axles', 'Number of axles'
]
scaler = StandardScaler()
train_df[scaling_columns] = scaler.fit_transform(train_df[scaling_columns])
train_df.describe()
sequences = list()

for name, group in tqdm(train_df.groupby(['ID'])):
    sequences.append(group.drop(columns=['ID']).values)
    
# train_values = np.asarray(train_values)
print(sequences[2].shape)
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
seq_len = 80
final_seq=tf.keras.preprocessing.sequence.pad_sequences(final_seq, maxlen=seq_len, padding='post', dtype='float', truncating='post')
# from keras.utils.np.utils import to_categorical
# y_train = to_categorical(y_train)

target = pd.get_dummies(train['DrivingStyle'])
target = np.asarray(target)
target.shape
train['DrivingStyle'].unique()
X_train,X_test,y_train,y_test = train_test_split(final_seq,target,test_size=0.20,random_state=34)
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
    final_seq,
    target,
    epochs=100,
    batch_size=84,
    validation_data=(X_test,y_test),
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(patience=5),
    ]
)

