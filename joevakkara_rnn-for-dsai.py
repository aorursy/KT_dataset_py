%matplotlib inline

import matplotlib.pyplot as plt

import tensorflow as tf

import numpy as np

import pandas as pd

import os

import datetime

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Input, Dense, GRU, Embedding

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from tensorflow.keras.backend import square, mean
tf.__version__
tf.keras.__version__
pd.__version__
!ls "../input/ca-data"
path = '../input/ca-data/'

ca1_data = pd.read_csv(path+"CA1_ext.csv")

ca2_data = pd.read_csv(path+"CA2_ext.csv")

ca3_data = pd.read_csv(path+"CA_3_ext.csv")

ca4_data = pd.read_csv(path+"CA4_ext.csv")

tx1_data = pd.read_csv(path+"TX_1_ext.csv")

tx2_data = pd.read_csv(path+"TX_2_ext.csv")

tx3_data = pd.read_csv(path+"TX_3_ext.csv")

wi1_data = pd.read_csv(path+"WI_1_ext.csv")

wi2_data = pd.read_csv(path+"WI_2_ext.csv")

wi3_data = pd.read_csv(path+"WI_3_ext.csv")

data = {}

data["CA_1"] = ca1_data[["Hobbie_revenue","House_revenue","Foods_revenue"]]

data["CA_2"] = ca2_data[["Hobbie_revenue","House_revenue","Foods_revenue"]]

data["CA_3"] = ca3_data[["Hobbie_revenue","House_revenue","Foods_revenue"]]

data["CA_4"] = ca4_data[["Hobbie_revenue","House_revenue","Foods_revenue"]]

data["TX_1"] = tx1_data[["Hobbie_revenue","House_revenue","Foods_revenue"]]

data["TX_2"] = tx2_data[["Hobbie_revenue","House_revenue","Foods_revenue"]]

data["TX_3"] = tx3_data[["Hobbie_revenue","House_revenue","Foods_revenue"]]

data["WI_1"] = wi1_data[["Hobbie_revenue","House_revenue","Foods_revenue"]]

data["WI_2"] = wi2_data[["Hobbie_revenue","House_revenue","Foods_revenue"]]

data["WI_3"] = wi3_data[["Hobbie_revenue","House_revenue","Foods_revenue"]]

data
data["CA_1"].head()
listofstore = ["CA_1","CA_2","CA_3","CA_4","TX_1","TX_2","TX_3","WI_1","WI_2","WI_3"]

listofstore
data_temp = data["CA_1"].join(data["CA_2"], lsuffix='_CA_1', rsuffix='_CA_2')

for store in listofstore[2:]:

    data_temp1 = data_temp.join(data[store], lsuffix='', rsuffix=store)

    data_temp1 = data_temp1.rename(columns={"Hobbie_revenue": "Hobbie_revenue_"+store,"House_revenue": "House_revenue_"+store,"Foods_revenue": "Foods_revenue_"+store})

    data_temp = data_temp1

    

#data_temp = data_temp1.join(data["TX_1"], lsuffix='', rsuffix='_TX_1')



data_df = data_temp1.copy()
data_df.head()
data_df.values.shape
import datetime

numdays = 1913

base = datetime.datetime(2011, 1, 29)

date_list = [base + datetime.timedelta(days=x) for x in range(numdays)]
from datetime import datetime

dayofyearlist = [i.timetuple().tm_yday for i in date_list]
data_df["Dayofyear"] = dayofyearlist
data_df
target_store = 'CA_1'
target_names = ['Hobbie_revenue', 'House_revenue', 'Foods_revenue']
shift_months = 1

shift_steps = shift_months * 30  # Number of days.
data_targets = data[target_store][target_names].shift(-shift_steps)
data[target_store][target_names].head(shift_steps + 5)
data_targets.head(5)
data_targets.tail()
data_df.values
x_data = data_df.values[0:-shift_steps]
print(type(x_data))

print("Shape:", x_data.shape)
y_data = data_targets.values[:-shift_steps]

y_data
print(type(y_data))

print("Shape:", y_data.shape)
num_data = len(x_data)

num_data
train_split = 0.9
num_train = int(train_split * num_data)

num_train
num_test = num_data - num_train

num_test
x_train = x_data[0:num_train]

x_test = x_data[num_train:]

len(x_train) + len(x_test)
y_train = y_data[0:num_train]

y_test = y_data[num_train:]

len(y_train) + len(y_test)
num_x_signals = x_data.shape[1]

num_x_signals
num_y_signals = y_data.shape[1]

num_y_signals
print("Min:", np.min(x_train))

print("Max:", np.max(x_train))
x_scaler = MinMaxScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
print("Min:", np.min(x_train_scaled))

print("Max:", np.max(x_train_scaled))
x_test_scaled = x_scaler.transform(x_test)
y_scaler = MinMaxScaler()

y_train_scaled = y_scaler.fit_transform(y_train)

y_test_scaled = y_scaler.transform(y_test)
print(x_train_scaled.shape)

print(y_train_scaled.shape)
def batch_generator(batch_size, sequence_length):

    """

    Generator function for creating random batches of training-data.

    """



    # Infinite loop.

    while True:

        # Allocate a new array for the batch of input-signals.

        x_shape = (batch_size, sequence_length, num_x_signals)

        x_batch = np.zeros(shape=x_shape, dtype=np.float16)



        # Allocate a new array for the batch of output-signals.

        y_shape = (batch_size, sequence_length, num_y_signals)

        y_batch = np.zeros(shape=y_shape, dtype=np.float16)



        # Fill the batch with random sequences of data.

        for i in range(batch_size):

            # Get a random start-index.

            # This points somewhere into the training-data.

            idx = np.random.randint(num_train - sequence_length)

            

            # Copy the sequences of data starting at this index.

            x_batch[i] = x_train_scaled[idx:idx+sequence_length]

            y_batch[i] = y_train_scaled[idx:idx+sequence_length]

        

        yield (x_batch, y_batch)
batch_size = 256
sequence_length = 30 * 6

sequence_length
generator = batch_generator(batch_size=batch_size,

                            sequence_length=sequence_length)
x_batch, y_batch = next(generator)
print(x_batch.shape)

print(y_batch.shape)
batch = 0   # First sequence in the batch.

signal = 0  # First signal from the 20 input-signals.

seq = x_batch[batch, :, signal]

plt.plot(seq)
seq = y_batch[batch, :, signal]

plt.plot(seq)
validation_data = (np.expand_dims(x_test_scaled, axis=0),

                   np.expand_dims(y_test_scaled, axis=0))
model = Sequential()
model.add(GRU(units=512,

              return_sequences=True,

              input_shape=(None, num_x_signals,)))
model.add(Dense(num_y_signals, activation='sigmoid'))
if False:

    from tensorflow.python.keras.initializers import RandomUniform



    # Maybe use lower init-ranges.

    init = RandomUniform(minval=-0.05, maxval=0.05)



    model.add(Dense(num_y_signals,

                    activation='linear',

                    kernel_initializer=init))
warmup_steps = 30
def loss_mse_warmup(y_true, y_pred):

    """

    Calculate the Mean Squared Error between y_true and y_pred,

    but ignore the beginning "warmup" part of the sequences.

    

    y_true is the desired output.

    y_pred is the model's output.

    """



    # The shape of both input tensors are:

    # [batch_size, sequence_length, num_y_signals].



    # Ignore the "warmup" parts of the sequences

    # by taking slices of the tensors.

    y_true_slice = y_true[:, warmup_steps:, :]

    y_pred_slice = y_pred[:, warmup_steps:, :]



    # These sliced tensors both have this shape:

    # [batch_size, sequence_length - warmup_steps, num_y_signals]



    # Calculat the Mean Squared Error and use it as loss.

    mse = mean(square(y_true_slice - y_pred_slice))

    

    return mse
optimizer = RMSprop(lr=1e-3)
model.compile(loss=loss_mse_warmup, optimizer=optimizer)
model.summary()
path_checkpoint = '23_checkpoint.keras'

callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,

                                      monitor='val_loss',

                                      verbose=1,

                                      save_weights_only=True,

                                      save_best_only=True)
callback_early_stopping = EarlyStopping(monitor='val_loss',

                                        patience=5, verbose=1)
callback_tensorboard = TensorBoard(log_dir='./23_logs/',

                                   histogram_freq=0,

                                   write_graph=False)
callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',

                                       factor=0.1,

                                       min_lr=1e-4,

                                       patience=0,

                                       verbose=1)
callbacks = [callback_early_stopping,

             callback_checkpoint,

             callback_tensorboard,

             callback_reduce_lr]
%%time

model.fit(x=generator,

          epochs=30,

          steps_per_epoch=100,

          validation_data=validation_data,

          callbacks=callbacks)
try:

    model.load_weights(path_checkpoint)

except Exception as error:

    print("Error trying to load checkpoint.")

    print(error)
result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),

                        y=np.expand_dims(y_test_scaled, axis=0))
print("loss (test-set):", result)
# If you have several metrics you can use this instead.

if False:

    for res, metric in zip(result, model.metrics_names):

        print("{0}: {1:.3e}".format(metric, res))
def plot_comparison(start_idx, length=100, train=True):

    """

    Plot the predicted and true output-signals.

    

    :param start_idx: Start-index for the time-series.

    :param length: Sequence-length to process and plot.

    :param train: Boolean whether to use training- or test-set.

    """

    

    if train:

        # Use training-data.

        x = x_train_scaled

        y_true = y_train

    else:

        # Use test-data.

        x = x_test_scaled

        y_true = y_test

    

    # End-index for the sequences.

    end_idx = start_idx + length

    

    # Select the sequences from the given start-index and

    # of the given length.

    x = x[start_idx:end_idx]

    y_true = y_true[start_idx:end_idx]

    

    # Input-signals for the model.

    x = np.expand_dims(x, axis=0)



    # Use the model to predict the output-signals.

    y_pred = model.predict(x)

    

    # The output of the model is between 0 and 1.

    # Do an inverse map to get it back to the scale

    # of the original data-set.

    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])

    

    #print(loss_mse_warmup(y_true, y_pred))

    

    # For each output-signal.

    for signal in range(len(target_names)):

        # Get the output-signal predicted by the model.

        signal_pred = y_pred_rescaled[:, signal]

        

        # Get the true output-signal from the data-set.

        signal_true = y_true[:, signal]

        



        # Make the plotting-canvas bigger.

        plt.figure(figsize=(15,5))

        

        # Plot and compare the two signals.

        plt.plot(signal_true, label='true')

        plt.plot(signal_pred, label='pred')

        

        # Plot grey box for warmup-period.

        p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)

        

        # Plot labels etc.

        plt.ylabel(target_names[signal])

        plt.legend()

        plt.show()
plot_comparison(start_idx=1000, length=500, train=True)
data["CA_4"]['Hobbie_revenue'][1000:1000+500].plot();
plot_comparison(start_idx=10, length=500, train=False)