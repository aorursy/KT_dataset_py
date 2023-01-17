# This code is adapted from https://blog.quantinsti.com/stock-market-data-analysis-python/

import pandas as pd

from pandas_datareader import data

from datetime import datetime



# Set the start and end date to match the entire history of the stock

start_date = '1994-01-01'

end_date = datetime.date(datetime.now())



# Set the ticker to get the Amazon stock info

ticker = 'AMZN'



# Fetch the data

stock_data = data.get_data_yahoo(ticker, start_date, end_date)



# Inspect the parameters and values of some of the data

stock_data.head()
# add a column that is the change between open and close as a normalization of the data

stock_data['Change'] = stock_data['Open'] - stock_data['Close']

stock_data.tail()
# create a new dataframe that contains the open to close difference of the last week (7 days)

seven_days_ago = stock_data['Change'][:-7].reset_index(drop=True)

six_days_ago   = stock_data['Change'][1:-6].reset_index(drop=True)

five_days_ago  = stock_data['Change'][2:-5].reset_index(drop=True)

four_days_ago  = stock_data['Change'][3:-4].reset_index(drop=True)

three_days_ago = stock_data['Change'][4:-3].reset_index(drop=True)

two_days_ago   = stock_data['Change'][5:-2].reset_index(drop=True)

one_day_ago    = stock_data['Change'][6:-1].reset_index(drop=True)

today          = stock_data['Change'][7:]



change_dataframe = pd.DataFrame({

    '7 days ago': seven_days_ago.values,

    '6 days ago': six_days_ago.values,

    '5 days ago': five_days_ago.values,

    '4 days ago': four_days_ago.values,

    '3 days ago': three_days_ago.values,

    '2 days ago': two_days_ago.values,

    '1 day ago': one_day_ago.values,

    'today': today.reset_index(drop=True).values,

    'date today' : today.index

})



# verify that the data is formated correctly

change_dataframe.tail(8)
# Import sklearn tools

from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score
# Split data into X and y

# X is the past change between market start and end

stock_X = change_dataframe[['7 days ago', '6 days ago', '5 days ago', '4 days ago', '3 days ago', '2 days ago', '1 day ago']]

# y is the change between market start and end today

stock_y = change_dataframe['today']



# Split the data into training and testing data (80/20 split)

stock_X_train = stock_X.iloc[:-278]

stock_X_test = stock_X.iloc[-278:]



stock_y_train = stock_y.iloc[:-278]

stock_y_test = stock_y.iloc[-278:]
# Fit the data using multiple regression

model = linear_model.LinearRegression()



model.fit(stock_X_train, stock_y_train)



print(model.coef_)

print(model.intercept_)
stock_y_pred = model.predict(stock_X_test)



print("Mean squared error between model prediction and true values: \n", mean_squared_error(stock_y_test, stock_y_pred))

print("Coefficient of determination of the regression model (prediction vs true values): \n", r2_score(stock_y_test, stock_y_pred))
import tensorflow as tf

import numpy as np
def univariate_data(dataset, start_index, end_index, history_size, target_size):

    '''

    Adapted from https://www.tensorflow.org/tutorials/structured_data/time_series

    Splits the data into a time chunk of the past (history_size) and the value of the next day.

    To

    PARAMETERS

    ----------

    dataset      : the column data to be chunked and split

    start_index  : the starting index to be used in the split

    end_index    : the ending index to be used in the split

    history_size : the size of the past chunk to be create for the model input

    target_size  : the time in the future to be predicted

    

    OUTPUT

    -----

    (split_x, split_y) : The x and y values for the training data. The dimensions of which depend on the history_size and target_size, as well as input indexes.

    '''

    data = []

    labels = []



    start_index = start_index + history_size

    if end_index is None:

        end_index = len(dataset) - target_size



    for i in range(start_index, end_index):

        indices = range(i-history_size, i)

        # Reshape data from (history_size,) to (history_size, 1)

        data.append(np.reshape(dataset[indices], (history_size, 1)))

        labels.append(dataset[i+target_size])

    return np.array(data), np.array(labels)
import math

# Normalize our data

LSTM_stock_data = (stock_data['Close'])/(stock_data['High']-stock_data['Low'])



LSTM_stock_data = LSTM_stock_data.values



# Split the data 80/20

TRAIN_SPLIT = math.floor(stock_data.shape[0] * .8)

# Working with the last 30 days

past_history = 30

# Only predicting today

future_target = 0



# Seperate into training and test data and shape the data for the model.

# This will split the data into the inputs (x) and the expected values (y). 

# x will contain rows of past data, and y will contain rows of the following day after x.

x_train, y_train = univariate_data(LSTM_stock_data, 0, TRAIN_SPLIT, past_history, future_target)

x_test,  y_test  = univariate_data(LSTM_stock_data, TRAIN_SPLIT, None, past_history, future_target)
# Prepare the data to be fed into the RNN

BATCH_SIZE = 256

BUFFER_SIZE = 500



# Chunk the data for better memory handling by the model. Too large of chunks will overload process memory and cause the trainig to fail.

train = tf.data.Dataset.from_tensor_slices((x_train, y_train))

train = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

test = test.batch(BATCH_SIZE).repeat()


with tf.device('/gpu:0'): # If working without a gpu, remove this line 

    '''

    Adapted from:

    https://towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model-4225457f0233#:~:text=Utilizing%20a%20Keras%20LSTM%20model%20to%20forecast%20stock%20trends&text=At%20the%20same%20time%2C%20these,LSTM)%20for%20times%20series%20forecasting.

    '''

    lstm_model = tf.keras.models.Sequential([

        # 50 LSTM layers with input width of the training data x values

        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=x_train.shape[-2:]),

        # A 20% drop out layer

        tf.keras.layers.Dropout(0.2),

        # 50 LSTM layers

        tf.keras.layers.LSTM(50),

        # A 20% drop out layer

        tf.keras.layers.Dropout(0.2),

        # Dense layer to reduce the output to only 1 value

        tf.keras.layers.Dense(1)

    ])



    lstm_model.compile(optimizer="adam", loss="mae")



# 200 data passes per epoch

EVALUATION_INTERVAL = 200

# 30 epochs

EPOCHS = 30



# Fit the model to the data

model_history = lstm_model.fit(train, epochs=EPOCHS,

         steps_per_epoch=EVALUATION_INTERVAL,

         validation_data=test, validation_steps=50)

import matplotlib.pyplot as plt



def plot_train_history(history, title):

    '''

    Adapted from: https://www.tensorflow.org/tutorials/structured_data/time_series

    Graphs the loss and validation loss of the model as it trains per epoch.

    

    PARAMETERS

    ----------

    history : a keras model history object. Contains the history data of the model's training process.

    title   : the title to place at the top of the figure.

    

    OUTPUT

    ------

    NONE

    '''

    loss = history.history['loss']

    val_loss = history.history['val_loss']



    epochs = range(len(loss))



    plt.figure()



    plt.plot(epochs, loss, 'b', label='Training loss')

    plt.plot(epochs, val_loss, 'r', label='Validation loss')

    plt.title(title)

    plt.legend()



    plt.show()
plot_train_history(model_history, 'Training History')
def create_time_steps(length):

    '''

    Adapted From https://www.tensorflow.org/tutorials/structured_data/time_series

    Generates an array of integers from -length to 0.

    

    PRAMETERS

    ---------

    length : integer that is the length of the range which is to be created

    

    OUTPUT

    ------

    time_step_list : a list of integers from -length to 0.

    '''

    return list(range(-length, 0))



def show_plot(plot_data, delta, title):

    '''

    Adapted from https://www.tensorflow.org/tutorials/structured_data/time_series

    Plots the past values of plot_data as a line graph, marks the final value of the time set, and marks the value of the model's prediction.

    

    PARAMETERS

    ----------

    plot_data : the data to be plotted as a dataframe or numpy array

    delta     : the distance into the future the model is expected to predict (defaults to 0)

    title     : the title of the figure as a string

    

    OUTPUT

    ------

    plt       : matplot lib plot object containing a figure of the past data values, the expected values over delta time, and the predicted values over delta time.

    '''

    labels = ['History', 'True Future', 'Model Prediction']

    marker = ['.-', 'rx', 'go']

    time_steps = create_time_steps(plot_data[0].shape[0])

    if delta:

        future = delta

    else:

        future = 0



    plt.title(title)

    for i, x in enumerate(plot_data):

        if i:

              plt.plot(future, plot_data[i], marker[i], markersize=10,

                   label=labels[i])

        else:

              plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])

    plt.legend()

    plt.xlim([time_steps[0], (future+5)*2])

    plt.xlabel('Time-Step')

    return plt
# Adapted from https://www.tensorflow.org/tutorials/structured_data/time_series

# Graph 10 predictions from our test data set

for x, y in test.take(10):

    plot = show_plot([x[0].numpy(), y[0].numpy(),

                      lstm_model.predict(x)[0]], 0, 'LSTM Stock Data Prediction')

    plot.show()