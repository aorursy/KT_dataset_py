import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from tqdm import tqdm_notebook

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import pickle

from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout

from keras.layers import LSTM

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

from keras import optimizers

from keras import backend as K

import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode(connected=True)

from plotly import tools

import plotly.figure_factory as ff

import os

import sys

import time
DATA_FOLDER = '../input/1-minute'

input_data = pd.read_csv(os.path.join(DATA_FOLDER, 'BTCUSDT.csv'), index_col='Time')

input_data.drop('Unnamed: 0', axis=1, inplace=True)

input_data.describe()
#convert the time to year-month-day format

input_data.index = pd.to_datetime(input_data.index, unit='ms')

input_data.sort_index(inplace=True)

input_data = input_data.resample('D').last()

input_data.head()
#check for nans

print(np.isnan(input_data).any())
# split the train and test

df_train, df_test = train_test_split(input_data, train_size=0.8, test_size=0.2, shuffle=False)

print("Train--Test size", len(df_train), len(df_test))



# scale the feature MinMax, build array

x = df_train.loc[:,df_train.columns].values

min_max_scaler = MinMaxScaler()

x_train = min_max_scaler.fit_transform(x)

x_test = min_max_scaler.transform(df_test.loc[:,df_train.columns])
def trim_dataset(data,batch_size):

    """

    trims dataset to a size that's divisible by BATCH_SIZE

    """

    no_of_rows_drop = data.shape[0]%batch_size

    if no_of_rows_drop > 0:

        return data[:-no_of_rows_drop]

    else:

        return data

    



def build_timeseries(data, y_col_index, TIME_STEPS):

    """

    Converts ndarray into timeseries fordata and supervised data fordata. Takes first TIME_STEPS

    number of rows as input and sets the TIME_STEPS+1th data as corresponding output and so on.

    :param data: ndarray which holds the dataset

    :param y_col_index: index of column which acts as output

    :return: returns two ndarrays-- input and output in fordata suitable to feed

    to LSTM.

    """

    # total number of time-series samples would be len(data) - TIME_STEPS

    dim_0 = data.shape[0] - TIME_STEPS

    dim_1 = data.shape[1]

    x = np.zeros((dim_0, TIME_STEPS, dim_1))

    y = np.zeros((dim_0,))

    print("dim_0",dim_0)

    for i in range(dim_0):

        x[i] = data[i:TIME_STEPS+i]

        y[i] = data[TIME_STEPS+i, y_col_index]

#         if i < 10:

#           print(i,"-->", x[i,-1,:], y[i])

    print("length of time-series i/o",x.shape,y.shape)

    return x, y

BATCH_SIZE = 4

params = {

    "epochs": 25,

    "lr": 0.0001,

}



OUTPUT_PATH = './'



def create_model(TIME_STEPS,layer1,layer2):

    lstm_model = Sequential()

    # (batch_size, timesteps, data_dim)

    lstm_model.add(LSTM(layer1, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]),

                        recurrent_activation='tanh', stateful=True, return_sequences=True, kernel_initializer='random_uniform'))

    lstm_model.add(Dropout(0.4))

    lstm_model.add(LSTM(layer2,recurrent_activation='tanh'))

    lstm_model.add(Dropout(0.4))

    lstm_model.add(Dense(25,activation='relu'))

    lstm_model.add(Dense(1,activation='sigmoid'))

    optimizer = optimizers.Adam(lr=params["lr"])

    lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)

    return lstm_model
def training(t,a,b):

    start_time = time.time()

    print("Building model...")

    model = create_model(t,a,b)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,

                       patience=40, min_delta=0.0001)



    mcp = ModelCheckpoint(os.path.join(OUTPUT_PATH,

                          "best_model.h5"), monitor='val_loss', verbose=1,

                          save_best_only=True, save_weights_only=False, mode='min', period=1)





    history = model.fit(x_t, y_t, epochs=params["epochs"], verbose=2, batch_size=BATCH_SIZE,

                        shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),

                        trim_dataset(y_val, BATCH_SIZE)), callbacks=[es, mcp])

    end_time = time.time()

    

    print("saving model... and total time for training "+str(int(end_time-start_time)))

    pickle.dump(model, open("lstm_model", "wb"))

    return history, model



def getError(x_test_t,y_test_t,model):

    y_pred = model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)

    y_pred = y_pred.flatten()

    y_test_t = trim_dataset(y_test_t, BATCH_SIZE)

    error = mean_squared_error(y_test_t, y_pred)

    return error
def combined(x,y,length):

    X = np.binary_repr(x,width=length)

    Y = np.binary_repr(y,width=length)

    Z = ''

    for i in range(len(X)):

        # Crossover happens with a probability of 50 %

        if(np.random.randint(4)>2):

            Z+=X[i]

        else:

            Z+=Y[i]

    output = int(Z,2)

    if(output==0):

        output = np.random.randint(1,2^length)

    return output



def crossover(currPopulation):

    a,b,c  = currPopulation

    A, B, C = [], [], []

    for i in range(4):

        for j in range(i+1,5):

            A.append(combined(a[i], a[j], MAX_TIME))

            B.append(combined(b[i], b[j], MAX_L1))

            C.append(combined(c[i], c[j], MAX_L2))

    return A, B, C



def flip(x,length):

    A = np.binary_repr(x,width=length)

    index = np.random.randint(length)    

    if(A[index]=='0'):

        x+=(2^index)

    else:

        x-=(2^index)

    output = x

    if(output==0):

        output = np.random.randint(1,2^length)

    return output



def mutation(currPopulation):

    # Crossover happens with a probability of 50 %

    if(np.random.randint(100)<2):

        selection_index = np.random.randint(0,2)

        lengths = [MAX_TIME, MAX_L1, MAX_L2]

        a = list(currPopulation[selection_index])

        index = np.random.randint(0,10)

        # Flip a bit in randomly selected population

        a[index] = flip(a[index], lengths[selection_index]) 

        currPopulation[selection_index] = a

    return currPopulation



def populationGenerator(currPopulation):

    if(len(currPopulation[0])==0):

        return [np.random.randint(1,2^MAX_TIME) for i in range(population_size)], [np.random.randint(1,2^MAX_L1) for i in range(population_size)], [np.random.randint(1,2^MAX_L2) for i in range(population_size)]

    crossedPopulation = crossover(currPopulation)

    finalPopulation = mutation(crossedPopulation)

    return finalPopulation





def getTop5(final_population):

    final_population.sort(key=lambda x:x[0])

    return  [a[1] for a in final_population[:5]], [a[2] for a in final_population[:5]], [a[3] for a in final_population[:5]]
currPopulation = [], [], []

population_size = 10

generation_count = 10

MAX_TIME, MAX_L1, MAX_L2 = 4, 7, 5





for generation in range(generation_count):

    print('---------------------------------\n============'+str(generation)+'============\n')

    t,l1, l2 = populationGenerator(currPopulation)

    final_list = []

    for a,b,c in zip(t,l1,l2):

        x_t, y_t = build_timeseries(x_train, 3, a)

        x_t = trim_dataset(x_t, BATCH_SIZE)

        y_t = trim_dataset(y_t, BATCH_SIZE)



        x_temp, y_temp = build_timeseries(x_test, 3, a)

        x_val, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE),2)

        y_val, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE),2)

        history, model = training(a,b,c)

        e = getError(x_test_t,y_test_t,model)

        final_list.append([1-e,a,b,c])

        del history, model

    currPopulation = getTop5(final_list)
a,b,c = currPopulation

print("The final window size "+str(a[0]) +" with first lstm layer of output size "+str(b[0])+" with second lstm layer of output size "+str(c[0]))
def plot_predictions(x_test_t,y_test_t):

    y_pred = model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)

    y_pred = y_pred.flatten()

    y_test_t = trim_dataset(y_test_t, BATCH_SIZE)

    error = mean_squared_error(y_test_t, y_pred)

    print("Error is", error, y_pred.shape, y_test_t.shape)



    y_pred_train = model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)

    # convert the predicted value to range of real data

    y_pred_org = (y_pred * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]

    # min_max_scaler.inverse_transform(y_pred)

    y_test_t_org = (y_test_t * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]

    return error

    # Visualize the prediction

    from matplotlib import pyplot as plt

    plt.figure()

    plt.plot(y_pred_org)

    plt.plot(y_test_t_org)

    plt.title('Prediction vs Real Stock Price')

    plt.ylabel('Price')

    plt.xlabel('Days')

    plt.legend(['Prediction', 'Real'], loc='upper left')

    plt.show()
# # Visualize the losses

# from matplotlib import pyplot as plt

# plt.figure()

# plt.plot(history.history['loss'])

# plt.plot(history.history['val_loss'])

# plt.title('Model loss')

# plt.ylabel('Loss')

# plt.xlabel('Epoch')

# plt.legend(['Train', 'Test'], loc='upper left')