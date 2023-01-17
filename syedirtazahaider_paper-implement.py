from pandas import read_csv

import numpy as np



# Load the data 

dataset_consider = read_csv('../input/Complete_8282282.csv', header=0, infer_datetime_format=True, 

                            parse_dates=['READING_DATETIME'], index_col=['READING_DATETIME'])
# convert dataframe to array

data_array = dataset_consider.values



# set lookback time

look_back = 2
# train, test split of data. 

# training data is from 01 June 2013 to 05 Aug 2013, total of 66 days

# validation data is from 06 Aug 2013 to 22 Aug 2013, total of 17 days

# test data is from 23 Aug 2013 to 31 Aug 2013, total of 9 days



train_split_end = ((66+17) * 48) - 1 - look_back



train = data_array[0:train_split_end,:]

test = data_array[train_split_end:,:]
# convert an array of values into a dataset matrix

def create_dataset(dataset, look_back):

	dataX, dataY = [], []

	for i in range(len(dataset)-look_back-1):

		a = dataset[i:(i+look_back), :]

		dataX.append(a)

		dataY.append(dataset[i + look_back, 0])

	return np.array(dataX), np.array(dataY)



x_train, y_train = create_dataset(train, look_back)

x_test, y_test = create_dataset(test, look_back)
# Float 64 to Float 32



x_train = np.float32(x_train)

y_train = np.float32(y_train)



x_test = np.float32(x_test)

y_test = np.float32(y_test)



# Reshape y

y_train = np.expand_dims(y_train, axis=1)

y_test = np.expand_dims(y_test, axis=1)



print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
# Apply minmax scaling on x_train and x_test. 

from sklearn.preprocessing import MinMaxScaler



scalers = {}

for i in range(x_train.shape[1]):

    scalers[i] = MinMaxScaler()

    x_train[:, i, :] = scalers[i].fit_transform(x_train[:, i, :]) 

    x_test[:, i, :] = scalers[i].transform(x_test[:, i, :]) 
# Apply minmax scaling on y_train 

scaler_y = MinMaxScaler()

y_train  = scaler_y.fit_transform(y_train) 
# callbacks for keras model

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

path_checkpoint = 'Checkpoint.keras'

callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,

                                      monitor='val_loss',

                                      verbose=1,

                                      save_weights_only=True,

                                      save_best_only=True)



callback_early_stopping = EarlyStopping(monitor='val_loss',

                                        patience=10, verbose=1)



callback_tensorboard = TensorBoard(log_dir='./Checkpoint/',

                                   histogram_freq=0,

                                   write_graph=False)





callbacks = [callback_early_stopping,

             callback_checkpoint,

             callback_tensorboard]
# LSTM model from the paper



from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM





model = Sequential()

model.add(LSTM(20, return_sequences=True, input_shape=(look_back,x_train.shape[2])))

model.add(LSTM(20, return_sequences=True))

model.add(LSTM(20, return_sequences=False))

model.add(Dense(1, activation = 'relu'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train, y_train, epochs=150, verbose=2, batch_size = 32, validation_split=0.20, shuffle=False, callbacks = callbacks)
# Load the last saved model to avoid overfitting

try:

    model.load_weights(path_checkpoint)

except Exception as error:

    print("Error trying to load checkpoint.")

    print(error)
# test the predictions

predictions = model.predict(x_test)



#invert the predictions 

inv_predictions = scaler_y.inverse_transform(predictions)
# calculate mape



def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



mape = mean_absolute_percentage_error(y_test, inv_predictions)

print(mape)