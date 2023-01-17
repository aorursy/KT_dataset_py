# Favorite lib :)

!pip install livelossplot
import numpy as np

import pandas as pd

from IPython.display import display

from pathlib import Path

from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing.sequence import TimeseriesGenerator

from livelossplot.keras import PlotLossesCallback



pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 50)

pd.set_option('display.width', 1000)



ROOT_DIR = Path("/kaggle/input/bike-sharing-demand")

TRAIN_DATA_PATH = ROOT_DIR / "train.csv"

TEST_DATA_PATH = ROOT_DIR / "test.csv"
def expanded_index_datetime_col(data):

    data = data.copy()

    data["hour"] = data.index.hour

    data["weekday"] = data.index.weekday

    data["day"] = data.index.day

    data["month"] = data.index.month

    data["year"] = data.index.year

    return data



def replaced_with_onehot_cols(data, col_names):

    data = data.copy()

    

    for col_name in col_names:

        one_hot = pd.get_dummies(data[col_name], prefix=col_name)

        data = data.join(one_hot)

        

        # Original column is not needed anymore

        del data[col_name]

    return data



def display_cols(df):

    print(f"Columns: ({len(df.columns)}) {list(df.columns)}")



# We want to predict "count" for last time step in the input, not for the next time step!

# Therefore we need to shift i1t by one step

# https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/

# y_train = train[["count"]].values

# y_train = np.roll(y_train, 1)
TRAIN_VAL = pd.read_csv(TRAIN_DATA_PATH, parse_dates=True, index_col="datetime")

display(TRAIN_VAL)
train_val = expanded_index_datetime_col(TRAIN_VAL)

train_val = replaced_with_onehot_cols(train_val, col_names=["season", "holiday", "workingday", "weather", "weekday", "month", "year"]) # "year"

train_val = train_val.drop(["casual", "registered"], axis=1)



display(train_val.head())

train_val.shape
train = train_val[train_val["day"] <= 16]

print("Days range: {}-{}".format(train["day"].min(), train["day"].max()))

train.shape
val = train_val[train_val["day"] > 16]



print("Days range: {}-{}".format(val["day"].min(), val["day"].max()))

display(val.head(0))

val.shape
def normalized_cols(df, scaler):

    df = df.copy()

    return pd.DataFrame(scaler.fit_transform(df.values), columns=df.columns, index=df.index)



scaler = MinMaxScaler()



x_train = normalized_cols(df=train.drop("count", axis=1), scaler=scaler)

y_train = normalized_cols(df=train[["count"]], scaler=scaler)



x_val = normalized_cols(df=val.drop("count", axis=1), scaler=scaler)

y_val = normalized_cols(df=val[["count"]], scaler=scaler)
fig = plt.figure(figsize=(16,9))

plt.plot(y_train.index, y_train['count'], 'blue', label='train')

plt.plot(y_val.index, y_val['count'], 'orange', label='val')

plt.legend()

plt.grid()
# x = [[1,1], [2,2], [3, 3], [4, 4],  [5,5], [1,1], [2,2], [3, 3], [4, 4],  [5,5], [6, 6]]

# y = [11,    12,    13,     14,      15,    111,   112,   113,    1114,    115,   116]

# g = TimeseriesGenerator(x, y, length=5, batch_size=1)

# gg = TimeseriesGenerator(x, y, length=2, batch_size=1)

# g[1]

# c = g + gg



def gen(X, Y):

    start_idx = 0

    window_size = 2 * 24 # 2 days (if there are no gaps)

     

    while start_idx + window_size <= X.size:

        # Each row contains feature set

        rows = X[start_idx:start_idx + window_size]

        

        # Remove "day" from features (helper column only)

        sequence_of_features = rows.drop("day", axis=1)

        

        # Value for last element in input sequence (NOT from the future)

        target = Y["count"][start_idx + window_size - 1]

        yield sequence_of_features.values, target

        

        

        current_day = X["day"][start_idx]

        if start_idx + window_size < len(X):

            last_day = X["day"][start_idx + window_size]

        else:

            # Prevent IndexError

            break

            

        # Don't allow for making sequences containing time gaps

        if current_day > last_day:

            # skip to next month if day number would decrease.

            start_idx += window_size

        else:

            # shift next yielded window by stride (offset)

            start_idx += 1
from tensorflow.keras.utils import Sequence #, to_categorical, plot_model



class BatchGenerator(Sequence):



    def __init__(self, xy_gen, batch_size=32):

        self.batch_size = batch_size

        self.xy = list(xy_gen)

        

    def __len__(self):

        """Returns number of batches"""

        return len(self.xy) // self.batch_size

    

    def __getitem__(self, idx):

        batch_range = range(idx * self.batch_size, (idx + 1) * self.batch_size)        

        batch_with_sequences_of_features = []

        batch_with_targets = []

        

        for seq_id in batch_range:

            seq_of_features, target = self.xy[seq_id]

            batch_with_sequences_of_features.append(seq_of_features)

            batch_with_targets.append(target)

        

        # Prepare randomized indexes for shuffling mini-batches

        indices = np.arange(self.batch_size)

        np.random.shuffle(indices)

        

        # Convert to numpy and shuffle

        batch_with_sequences_of_features = np.array(batch_with_sequences_of_features)[indices]

        batch_with_targets = np.array(batch_with_targets)[indices]

        

        return batch_with_sequences_of_features, batch_with_targets
train_gen = BatchGenerator(gen(x_train, y_train), batch_size=64)

val_gen = BatchGenerator(gen(x_val, y_val), batch_size=64)

len(train_gen), len(val_gen)
# seq_length = 1 #24 * 3 # 2 days

# num_features = x_train[0].size

# batch_size = 64



# train_generator = TimeseriesGenerator(x_train, y_train, length=seq_length, batch_size=batch_size)

# print(f"Lookback: {seq_length} | Number of features: {num_features}")



# for idx in range(len(train_generator)):

#     print(train_generator[idx])

# train_generator[3]
# train_generator[1]

# int(len(train)/(19*24))
from keras.layers import Input, LSTM, Dense, Dropout, SimpleRNN

from keras.models import Model

from keras.callbacks import EarlyStopping

import keras.backend as K



def rmse(y_true, y_pred):

    """ root_mean_squared_error """

    return K.sqrt(K.mean(K.square(y_pred - y_true)))



input = Input(shape=(48, 38))

# _ = SimpleRNN(4, activation='relu')(input)

_ = LSTM(32, activation='relu')(input)

_ = Dropout(0.4)(_)

_ = Dense(16, activation='relu')(_)

_ = Dropout(0.4)(_)

_ = Dense(8, activation='relu')(_)

output = Dense(1, activation='relu')(_)



model = Model(inputs=input, outputs=output)

model.compile(optimizer='adam', loss='mse', metrics=[rmse])



history = model.fit_generator(train_gen, validation_data=val_gen,

                              epochs=25, verbose=1, 

                              callbacks=[	

                                    EarlyStopping(monitor="val_loss"),

                                    PlotLossesCallback(), 

                              ]

                             )
loss = history.history['loss']

loss_val = history.history['val_loss']

epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'bo', label='loss_train')

plt.plot(epochs, loss_val, 'b', label='loss_val')

plt.title('value of the loss function')

plt.xlabel('epochs')

plt.ylabel('value of the loss function')

plt.legend()

plt.grid()

plt.show()



# data.info()

# data[['temp', 'casual', 'registered']].plot(alpha=0.5)

# data.describe().drop("count")

# In case there would be some columns with null values: df['foo’].fillna(value=df['foo'].mean())
sequence_of_features = x_val[0:48].drop("day", axis=1)

pred = model.predict(np.expand_dims(sequence_of_features.values))



plt.plot(np.squeeze(pred), 'b', label='val pred')

plt.plot(np.squeeze(y_val["count"][0:48]), 'r', label='val gt')



# pred


model.predict(np.expand_dims(x_test, axis=1))

# split datetime column into year, day, month, hour

# use minmax scaler

# split 19-day train set into train and val (last x days)

# onehot - get dummies

# build generator

# prepare simple model

# prepare callbacks: livelossplot, earlystopping, etc.

# fit model
df['hour'] = df.index.hour #create column containing the hour

df['dayofweek'] = df.index.dayofweek
pd.read_csv(TEST_DATA_PATH)
from matplotlib import pyplot as plt

# plot each column

plt.figure()

plt.subplot(3, 1, 1)

plt.plot(data["temp"], 1, 1)

plt.subplot(3, 1, 2)

plt.plot(data["windspeed"], 1, 2)

plt.subplot(3,1,3)

plt.plot(data["casual"], 1, 3)

plt.show()