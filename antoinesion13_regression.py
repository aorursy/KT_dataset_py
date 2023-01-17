# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install pmdarima

!pip install keras



import keras

import tensorflow

import tensorflow.keras

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm

from pmdarima import auto_arima



# For NEURAL NETWORK

from keras import regularizers

from keras.models import Model, Sequential

from keras.layers import Dense

from keras.callbacks import EarlyStopping, ModelCheckpoint



# For ML

from sklearn.preprocessing import MinMaxScaler



import keras.backend as K

#from utils import keyvalue, embedding, smape

#import utils

#!pylint utils



def embedding(data, p):

    data_shifted = data.copy()

    for lag in range(-p+1, 2):

        data_shifted['y_t' + '{0:+}'.format(lag)] = data_shifted['y'].shift(-lag, freq='D')

    data_shifted = data_shifted.dropna(how='any')

    y = data_shifted['y_t+1'].to_numpy()

    X = data_shifted[['y_t' + '{0:+}'.format(lag) for lag in range(-p+1, 1)]].to_numpy()

    return (X,y, data_shifted)



def smape(y_true, y_pred):

    denominator = (y_true + K.abs(y_pred)) / 200.0

    diff = K.abs(y_true - y_pred) / denominator

    return K.mean(diff)



def keyvalue(df):

    df["horizon"] = range(1, df.shape[0]+1)

    res = pd.melt(df, id_vars = ["horizon"])

    res = res.rename(columns={"variable": "series"})

    res["Id"] = res.apply(lambda row: "s" + str(row["series"].split("-")[1]) + "h"+ str(row["horizon"]), axis=1)

    res = res.drop(['series', 'horizon'], axis=1)

    res = res[["Id", "value"]]

    res = res.rename(columns={"value": "forecasts"})

    return res



class TimeSeriesTensor(UserDict):

    """A dictionary of tensors for input into the RNN model.

    

    Use this class to:

      1. Shift the values of the time series to create a Pandas dataframe containing all the data

         for a single training example

      2. Discard any samples with missing values

      3. Transform this Pandas dataframe into a numpy array of shape 

         (samples, time steps, features) for input into Keras

    The class takes the following parameters:

       - **dataset**: original time series

       - **target** name of the target column

       - **H**: the forecast horizon

       - **tensor_structures**: a dictionary discribing the tensor structure of the form

             { 'tensor_name' : (range(max_backward_shift, max_forward_shift), [feature, feature, ...] ) }

             if features are non-sequential and should not be shifted, use the form

             { 'tensor_name' : (None, [feature, feature, ...])}

       - **freq**: time series frequency (default 'H' - hourly)

       - **drop_incomplete**: (Boolean) whether to drop incomplete samples (default True)

    """

    

    def __init__(self, dataset, target, H, tensor_structure, freq='H', drop_incomplete=True):

        self.dataset = dataset

        self.target = target

        self.tensor_structure = tensor_structure

        self.tensor_names = list(tensor_structure.keys())

        

        self.dataframe = self._shift_data(H, freq, drop_incomplete)

        self.data = self._df2tensors(self.dataframe)

    

    def _shift_data(self, H, freq, drop_incomplete):

        

        # Use the tensor_structures definitions to shift the features in the original dataset.

        # The result is a Pandas dataframe with multi-index columns in the hierarchy

        #     tensor - the name of the input tensor

        #     feature - the input feature to be shifted

        #     time step - the time step for the RNN in which the data is input. These labels

        #         are centred on time t. the forecast creation time

        df = self.dataset.copy()

        

        idx_tuples = []

        for t in range(1, H+1):

            df['t+'+str(t)] = df[self.target].shift(t*-1, freq=freq)

            idx_tuples.append(('target', 'y', 't+'+str(t)))



        for name, structure in self.tensor_structure.items():

            rng = structure[0]

            dataset_cols = structure[1]

            

            for col in dataset_cols:

            

            # do not shift non-sequential 'static' features

                if rng is None:

                    df['context_'+col] = df[col]

                    idx_tuples.append((name, col, 'static'))



                else:

                    for t in rng:

                        sign = '+' if t > 0 else ''

                        shift = str(t) if t != 0 else ''

                        period = 't'+sign+shift

                        shifted_col = name+'_'+col+'_'+period

                        df[shifted_col] = df[col].shift(t*-1, freq=freq)

                        idx_tuples.append((name, col, period))

                

        df = df.drop(self.dataset.columns, axis=1)

        idx = pd.MultiIndex.from_tuples(idx_tuples, names=['tensor', 'feature', 'time step'])

        df.columns = idx



        if drop_incomplete:

            df = df.dropna(how='any')



        return df

    

    def _df2tensors(self, dataframe):

        

        # Transform the shifted Pandas dataframe into the multidimensional numpy arrays. These

        # arrays can be used to input into the keras model and can be accessed by tensor name.

        # For example, for a TimeSeriesTensor object named "model_inputs" and a tensor named

        # "target", the input tensor can be acccessed with model_inputs['target']

    

        inputs = {}

        y = dataframe['target']

        y = y.as_matrix()

        inputs['target'] = y



        for name, structure in self.tensor_structure.items():

            rng = structure[0]

            cols = structure[1]

            tensor = dataframe[name][cols].as_matrix()

            if rng is None:

                tensor = tensor.reshape(tensor.shape[0], len(cols))

            else:

                tensor = tensor.reshape(tensor.shape[0], len(cols), len(rng))

                tensor = np.transpose(tensor, axes=[0, 2, 1])

            inputs[name] = tensor



        return inputs

       

    def subset_data(self, new_dataframe):

        

        # Use this function to recreate the input tensors if the shifted dataframe

        # has been filtered.

        

        self.dataframe = new_dataframe

        self.data = self._df2tensors(self.dataframe)
# Any results you write to the current directory are saved as output.

data = pd.read_csv("/kaggle/input/hands-on-ai-umons-2019/train.csv", index_col = "Day")





data.index = pd.to_datetime(data.index, format = "%Y-%m-%d")

data = data.asfreq('d')



interval_train = pd.date_range(start = '2015-07-01', end = '2017-08-09')

interval_valid  = pd.date_range(start = '2017-08-10',  end = '2017-08-20')



interval_test  = pd.date_range(start = '2017-08-21',  end = '2017-09-10')

HORIZON = len(interval_test)





data_train = data.loc[interval_train]

data_valid = data.loc[interval_valid]



forecasts_nn = pd.DataFrame(index = interval_test)
final_train_loss_mean = 0

final_val_loss_mean = 0

    



for iseries in data_train.columns:

    print(iseries)

    series_train = data_train[iseries]

    

    train = data_train[iseries].to_frame()

    valid = data_valid[iseries].to_frame()

    

    #####  MLP (recursive)

    LATENT_DIM = 5 # number of units in the dense layer

    BATCH_SIZE = 32 # number of samples per mini-batch

    EPOCHS = 100 # maximum number of times the training algorithm will cycle through all samples

    

    p = 3 #number of days used to predict 

    OUTPUT_LENGTH = 1



    scaler = MinMaxScaler()

    

    train["y"] = scaler.fit_transform(train)

    X_train, y_train, train_embedded = embedding(train, p)



    valid['y'] = scaler.transform(valid)

    X_valid, y_valid, valid_embedded  = embedding(valid, p)



    model = Sequential()

    model.add(Dense(LATENT_DIM, activation="relu", input_shape=(p,)))

    model.add(Dense(OUTPUT_LENGTH))

    model.compile(optimizer='RMSprop', loss='mse')

        

    best_val = ModelCheckpoint('model_{epoch:02d}.h5', save_best_only=True, mode='min', period=1)

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)

    history = model.fit(X_train,

                    y_train,

                    batch_size=BATCH_SIZE,

                    epochs=EPOCHS,

                    validation_data=(X_valid, y_valid),

                    callbacks=[earlystop, best_val],

                    verbose=1)

    

    best_val = ModelCheckpoint('model_{epoch:02d}.h5', save_best_only=True, mode='min', period=1)

    best_epoch = np.argmin(np.array(history.history['val_loss']))+1

    model.load_weights("model_{:02d}.h5".format(best_epoch))

    plot_df = pd.DataFrame.from_dict({'train_loss':history.history['loss'], 'val_loss':history.history['val_loss']})

    

    

    final_train_loss_mean += plot_df.train_loss.iloc[-1]

    

   

    final_val_loss_mean += plot_df.val_loss.iloc[-1]

    

    plot_df.plot(logy=True, figsize=(10,10), fontsize=12)

    plt.xlabel('epoch', fontsize=12)

    plt.ylabel('loss', fontsize=12)

    plt.show()

    

    # recursive forecasts

    x = X_valid[-1, 1:]

    ypred = y_valid[-1]

     

    f_nn = []

    for horizon in range(1, HORIZON+1):

        x_test = np.expand_dims(np.append(x, ypred), axis=0)

        ypred = model.predict(x_test)

        x = x_test[x_test.shape[0] - 1, 1:]

        f_nn.append(float(ypred))

    

    f_nn = np.expand_dims(f_nn, axis = 0)

    forecasts_nn[iseries] = scaler.inverse_transform(f_nn).flatten()

    

final_train_loss_mean = final_train_loss_mean / len(data_train.columns)

final_val_loss_mean = final_val_loss_mean / len(data_train.columns)

print(final_train_loss_mean)

print(final_val_loss_mean)

    

pred_naive = keyvalue(forecasts_nn)

pred_naive.to_csv("recursive.csv", index = False)
final_train_loss_mean = 0

final_val_loss_mean = 0

    







for iseries in data_train.columns:

    print(iseries)

    series_train = data_train[iseries]

    

    train = data_train[iseries].to_frame()

    valid = data_valid[iseries].to_frame()

    

    #####  MLP (recursive)

    LATENT_DIM = 10 # number of units in the dense layer

    BATCH_SIZE = 32 # number of samples per mini-batch

    EPOCHS = 100 # maximum number of times the training algorithm will cycle through all samples

    

    p = 60 #number of days used to predict 

    OUTPUT_LENGTH = HORIZON



    scaler = MinMaxScaler()

    

    train[[iseries]] = scaler.fit_transform(train)

    tensor_structure = {'X':(range(-p+1, 1), [iseries])}

    train_inputs = TimeSeriesTensor(train, iseries , HORIZON, tensor_structure, 'D')

    X_train = train_inputs.dataframe.as_matrix()[:,HORIZON:]

        

    valid[[iseries]] = scaler.fit_transform(valid)

    tensor_structure = {'X':(range(-p+1, 1), [iseries])}

    valid_inputs = TimeSeriesTensor(valid, iseries , HORIZON, tensor_structure, 'D')

    X_valid = valid_inputs.dataframe.as_matrix()[:,HORIZON:]





    model = Sequential()

    model.add(Dense(LATENT_DIM, activation="relu", input_shape=(p,)))

    model.add(Dense(OUTPUT_LENGTH))

    model.compile(optimizer='RMSprop', loss='mse')

        

    best_val = ModelCheckpoint('model_{epoch:02d}.h5', save_best_only=True, mode='min', period=1)

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)

    history = model.fit(X_train,

                    train_inputs['target'],

                    batch_size=BATCH_SIZE,

                    epochs=EPOCHS,

                    validation_data=(X_valid, valid_inputs['target']),

                    callbacks=[earlystop, best_val],

                    verbose=1)

    

    best_val = ModelCheckpoint('model_{epoch:02d}.h5', save_best_only=True, mode='min', period=1)

    best_epoch = np.argmin(np.array(history.history['val_loss']))+1

    model.load_weights("model_{:02d}.h5".format(best_epoch))

    plot_df = pd.DataFrame.from_dict({'train_loss':history.history['loss'], 'val_loss':history.history['val_loss']})

    

    

    final_train_loss_mean += plot_df.train_loss.iloc[-1]

    

   

    final_val_loss_mean += plot_df.val_loss.iloc[-1]

    

    plot_df.plot(logy=True, figsize=(10,10), fontsize=12)

    plt.xlabel('epoch', fontsize=12)

    plt.ylabel('loss', fontsize=12)

    plt.show()

    

    f_nn = []

    print(X_valid[-1].shape)

    xtest = np.expand_dims(X_valid[-1],axis=0)

    ypred = model.predict(xtest)

    f_nn = ypred

    forecasts_nn[iseries] = scaler.inverse_transform(f_nn).flatten()

    

final_train_loss_mean = final_train_loss_mean / len(data_train.columns)

final_val_loss_mean = final_val_loss_mean / len(data_train.columns)

print(final_train_loss_mean)

print(final_val_loss_mean)

    

pred_naive = keyvalue(forecasts_nn)

pred_naive.to_csv("multioutput.csv", index = False)