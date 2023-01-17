from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from plotly.subplots import make_subplots
import tensorflow.keras.layers as L
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import tensorflow as tf
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import os

data_dir = '../input/vehicle-data/'
# reading all data
vehicle = pd.read_csv(os.path.join(data_dir, 'Train_Vehicletravellingdata.csv'))
weather = pd.read_csv(os.path.join(data_dir, 'Train_WeatherData.csv'))
train = pd.read_csv(os.path.join(data_dir, 'Train.csv'))
vehicle.head()
weather.head()
train.head()
cons_df = pd.concat([vehicle, weather.drop(columns=['ID', 'Date time'])], axis=1)
cons_df = cons_df.merge(train, on=['ID'])
# Let's see the count of target variable
driving_style_df = cons_df['DrivingStyle'].value_counts().to_frame()
driving_style_df.columns = ['count']
driving_style_df['Labels'] = driving_style_df.index

fig = px.bar(driving_style_df, x='Labels', y='count')
fig.show()
# Count of missing values in columns
missing_values = cons_df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values()

missing_values = missing_values.to_frame()
missing_values.columns = ['count']
missing_values.index.names = ['Name']
missing_values['Name'] = missing_values.index

fig = px.bar(missing_values, x='Name', y='count')
fig.show()
# Checking distribution of Speed
df = px.data.tips()
fig = px.box(cons_df, y="Speed of the vehicle (kph)", points="all")
fig.show()
# Histogram
df = px.data.tips()
fig = px.histogram(cons_df, x="Speed of the vehicle (kph)", color="Lane of the road", marginal="violin")
fig.show()
# Aggregating on vehicle ID
agg_df = cons_df.groupby(['ID']).agg({'Speed of the vehicle (kph)': 'mean', 'DrivingStyle': 'first'})
# Histogram
df = px.data.tips()
fig = px.histogram(agg_df, x="Speed of the vehicle (kph)", color="DrivingStyle", marginal="violin")
fig.show()
transform_df = cons_df.copy()
transform_df['count'] = cons_df.groupby(['ID'])['DrivingStyle'].transform('count')
# Taking the vehicle of each class with maximum number of observations
ds1 = transform_df.loc[transform_df['DrivingStyle'] == 1]
ds1 = ds1.loc[ds1['count'] == ds1['count'].max()]
print(ds1['ID'].nunique())

ds2 = transform_df.loc[transform_df['DrivingStyle'] == 2]
ds2 = ds2.loc[ds2['count'] == ds2['count'].max()]
print(ds2['ID'].nunique())

ds3 = transform_df.loc[transform_df['DrivingStyle'] == 3]
ds3 = ds3.loc[ds3['count'] == ds3['count'].max()]
print(ds3['ID'].nunique())
ds1x = np.linspace(0, 1, len(ds1))
ds2x = np.linspace(0, 1, len(ds2))
ds3x = np.linspace(0, 1, len(ds3))

ds1y = ds1['Speed of the vehicle (kph)'].values
ds2y = ds2['Speed of the vehicle (kph)'].values
ds3y = ds3['Speed of the vehicle (kph)'].values

fig = go.Figure()
fig.add_trace(go.Scatter(x=ds1x, y=ds1y,
                    mode='lines',
                    name='aggressive'))
fig.add_trace(go.Scatter(x=ds2x, y=ds2y,
                    mode='lines',
                    name='normal'))
fig.add_trace(go.Scatter(x=ds3x, y=ds3y,
                    mode='lines', name='vague'))

fig.show()
label_encoding_columns = ['Road Condition', 'Precipitation', 'Precipitation intensity', 'Day time']
cons_df[label_encoding_columns] = cons_df[label_encoding_columns].apply(LabelEncoder().fit_transform)
train_df = cons_df.drop(columns=['ID of the preceding vehicle', 'Date time', 'DrivingStyle'])
missing_values_columns = list(missing_values['Name'])

for col in missing_values_columns:
    train_df[col] = train_df[col].fillna(train_df[col].mean())
train_df = train_df.drop(train_df[(train_df['Speed of the vehicle (kph)'] >140) | (train_df['Speed of the vehicle (kph)']< 20)].index).reset_index(drop=True)
# Speed Ratio
train_df['speed_ratio'] = (train_df['Speed of the vehicle (kph)']/train_df['Speed of the preceding vehicle']).clip(0, 2)

# Time ratio
train_df['time_ratio1'] = (train_df['Time gap with the preceeding vehicle in seconds']/train_df['Speed of the vehicle (kph)']).clip(0, 2)
train_df['time_ratio2'] = (train_df['Time gap with the preceeding vehicle in seconds']/train_df['Speed of the preceding vehicle']).clip(0, 2)

# Weight Ratio
train_df['weight_ratio'] = (train_df['Weight of the preceding vehicle']/train_df['weight of vehicle in kg']).clip(0, 100)

train_df.filter(regex=r'_ratio').describe()
scaling_columns = [
    'Speed of the vehicle (kph)', 'Speed of the preceding vehicle', 'Length of preceding vehicle',
    'Time gap with the preceeding vehicle in seconds', 'Weather details-Air temperature', 'Weight of the preceding vehicle',
    'Relative humidity', 'Wind direction', 'Wind speed in m/s', 'Length of vehicle in cm', 'weight of vehicle in kg',
    'Number of axles', 'Number of axles'
]
scaler = StandardScaler()
train_df[scaling_columns] = scaler.fit_transform(train_df[scaling_columns])
train_df.head()
train_df.columns
sequences = list()

for name, group in tqdm(train_df.groupby(['ID'])):
    sequences.append(group.drop(columns=['ID']).values)
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
   
    to_concat = np.repeat(one_seq[-1], n).reshape(21, n).transpose()
    new_one_seq = np.concatenate([one_seq, to_concat])
    new_seq.append(new_one_seq)
final_seq = np.stack(new_seq)

seq_len = 80
final_seq=tf.keras.preprocessing.sequence.pad_sequences(final_seq, maxlen=seq_len, padding='post', dtype='float', truncating='post')
target = pd.get_dummies(train['DrivingStyle'])
target = np.asarray(target)
X_train,X_test,y_train,y_test = train_test_split(final_seq,target,test_size=0.20,random_state=34)
model = tf.keras.models.Sequential()
model.add(L.LSTM(128, dropout=0.2, input_shape=(seq_len, 21), return_sequences=True))
model.add(L.LSTM(64, dropout=0.2, input_shape=(seq_len, 21), return_sequences=True))
model.add(L.LSTM(64, dropout=0.2))
model.add(L.Dense(3, activation='softmax'))
adam = tf.keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.summary()
history = model.fit(
    final_seq,
    target,
    epochs=150,
    batch_size=84,
    validation_data=(X_test,y_test),
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(patience=5),
    ]
)
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Loss", "Accuracy"),
    shared_yaxes=True,
    shared_xaxes=True,
    vertical_spacing=0.1,
    horizontal_spacing=0.03)

epochs = list(range(1, len(history.history['loss'])+1))

fig.add_trace(
    go.Scatter(x=epochs, y=history.history['loss'], name='loss'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=epochs, y=history.history['val_loss'], name='val_loss'),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=epochs, y=history.history['accuracy'], name='acc'),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(x=epochs, y=history.history['val_accuracy'], name='val_acc'),
    row=1, col=2
)