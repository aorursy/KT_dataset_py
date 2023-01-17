import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
dataPath = '../input'
setNumber = 1
id_col = ['id']

cycle_col = ['cycle']

setting_cols = ['setting1', 'setting2', 'setting3']

sensor_cols = ['sensor' + str(i) for i in range(1, 22)]

rul_col = ['RUL']

all_cols = id_col + cycle_col + setting_cols + sensor_cols + rul_col
# This section is to load data

def loadData(fileName):

    data = pd.read_csv(fileName, sep=" ", header=None)

    data.drop([26, 27], axis = 1, inplace=True)

    data.columns = id_col + cycle_col + setting_cols +sensor_cols

    return data
# load train RUL also returns the max cycle, and this max cycle is also the life cylce

def addTrainRul(data, decrease_threshold=None):

    lifeCycles = {mcId: data[data['id']==mcId]['cycle'].max() for mcId in data['id'].unique()}

    if decrease_threshold == None: decrease_threshold = 1

    ruls = [lifeCycles[row[0]] - decrease_threshold if row[1] < decrease_threshold else lifeCycles[row[0]] - row[1] for row in data.values]

    data['RUL'] = ruls

    return lifeCycles

    

# use this last one only, return the data as well as the max life cycles

def loadTrainData(setNumber, decrease_threshold=None):

    fileName = dataPath + '/train_FD00' + str(setNumber) + '.txt'

    data = loadData(fileName)

    lifeCycles = addTrainRul(data, decrease_threshold)

    return data, lifeCycles
decrease_threshold = None

train, trainLifeCycles = loadTrainData(setNumber, decrease_threshold)
def loadTestRul(fileName):

    data = pd.read_csv(fileName, sep = " ", header=None)

    data.drop([1], axis=1, inplace=True)

    data.columns = ['RUL']

    return data

def addTestRul(data, rulData, decrease_threshold=None):

    testRuls = {i+1: rulData.iloc[i, 0] for i in range(len(rulData))}

    lifeCycles = {mcId: data[data['id']==mcId]['cycle'].max() + testRuls[mcId] for mcId in data['id'].unique()}

    if decrease_threshold == None: decrease_threshold = 1

    ruls = [lifeCycles[row[0]] - decrease_threshold if row[1] < decrease_threshold else lifeCycles[row[0]] - row[1] for row in data.values]

    data['RUL'] = ruls

    return lifeCycles

# Use this last one only => return data as well as the max life cycles for each machine

def loadTestData(setNumber, decrease_threshold=None):

    data = loadData(dataPath + '/test_FD00' +str(setNumber)+'.txt')

    rulData = loadTestRul(dataPath + '/RUL_FD00' + str(setNumber)+'.txt')

    lifeCycles = addTestRul(data, rulData, decrease_threshold)

    return data, lifeCycles
# Also make test RUL becomes piecewise

test, testLifeCycles = loadTestData(setNumber, decrease_threshold)
from matplotlib import pyplot as plt

import seaborn as sns
# As of feature selection they often select: 7, 8, 9, 12, 16, 17, 20  (manual selection based on sensor trends)

def plotSensorDataOfId(data, mcId):

    plt.figure(figsize=(30, 20))

    for i in range(21):

        sensor = 'sensor'+str(i+1)

        plt.subplot(10, 3, i+1).set_title(sensor)

        ssdata = data[data['id']==mcId]

        plt.plot(ssdata['cycle'], ssdata[sensor])

    plt.tight_layout()
plotSensorDataOfId(train, 1)
def plotDataDistribution(data):

    sensors = []

    plt.figure(figsize=(30, 10))

    for i in range(21):

        sensor = 'sensor'+str(i+1)

        if(len(data[sensor].unique())>1):

            sensors.append(sensor)

            plt.subplot(3, 10, i+1)

            sns.distplot(data[sensor])

    plt.tight_layout()

    return sensors
# As of feature selection they often select: 7, 8, 9, 12, 16, 17, 20 => Why 16?

cols = plotDataDistribution(train)
def plotCorrelation(data):

    corr = data.corr()

    # Generate a mask for the upper triangle

    mask = np.zeros_like(corr, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=(10, 10))

    sns.heatmap(data.corr(), square=True, mask=mask, cbar_kws={"shrink": 0.5})
plotCorrelation(train[cols])
plt.scatter(train['sensor15'].values, train['sensor14'].values)
def plotCorrelationOfID(data, mcId):

    data1 = data[data['id']==mcId]

    data1 = data1.drop(['id'], axis = 1)

    corr = data1.corr()

    # Generate a mask for the upper triangle

    mask = np.zeros_like(corr, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=(10, 10))

    sns.heatmap(data1.corr(), square=True, mask=mask, cbar_kws={"shrink": 0.5})
plotCorrelationOfID(train[['id']+cols], 1)
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler
# Scale the data and return the scaled data in form of a df and the scaler (will generate the scaler if doesn't pass it)

def scaleData(data, scaler=None):

    scaled_fields = setting_cols+sensor_cols

    if scaler == None:

        scaler = StandardScaler().fit(data[scaled_fields].values)

#         scaler = MinMaxScaler().fit(data[scaled_fields].values)

    scaled_data = scaler.transform(data[scaled_fields].values)

    scaled_df0 = pd.DataFrame(scaled_data)

    scaled_df0.columns = scaled_fields

    scaled_df1 = data.copy()

    for i in range(len(scaled_fields)):

        theField = scaled_fields[i]

        scaled_df1[theField] = scaled_df0[theField]

    return scaled_df1, scaler
# Scaled train

scaled_train, scaler = scaleData(train)

# Scaled test

scaled_test, scaler = scaleData(test, scaler)
# plot to make sure that the scaled data still keep its shape.

cols = plotDataDistribution(scaled_train)
#plot to see if the data keeps its distribution

cols = plotDataDistribution(scaled_test)
plotSensorDataOfId(scaled_train, 1)
import random

def getPieceWiseData(data, augmentStartCycle=None, augmentEndCycle=None, movingAverage=None):

    uniqueIds = data['id'].unique()

    if movingAverage==None:

        result = [data[data['id']==mId].values for mId in uniqueIds]

    else:

        result = [data[data['id']==mId].rolling(movingAverage).mean().dropna().values for mId in uniqueIds]

    maxlen = np.max([len(x) for x in result])

    #Augment the data now

    if(augmentStartCycle!=None and augmentEndCycle!= None):

        result1 = []

        for mc in result:

            maxCycle = len(mc)

            for i in range(50):

                idx = random.randint(max([maxCycle-145, 10]), max([maxCycle-10, 10]))

                if(len(mc[:idx, :])>0):

                    result1.append(mc[:idx, :])

            #Also add the complete sequence.

#             result1.append(mc)

        result = result1

    # calculate the ruls (-1) is the last column for RUL

    ruls = [min(mc[:, -1]) for mc in result]

    return result,ruls, maxlen

# Use this last one only (prev one is a helper)

from keras.preprocessing.sequence import pad_sequences

def getPaddedSequence(data, pad_type='pre', maxlen=None, augmentStartCycle=None, augmentEndCycle=None, movingAverage=None):

    piece_wise, ruls, ml = getPieceWiseData(data, augmentStartCycle, augmentEndCycle, movingAverage)

    if(maxlen==None): maxlen = ml

    padded_sequence = pad_sequences(piece_wise, padding=pad_type, maxlen=maxlen, dtype='float32')

    return padded_sequence, ruls, maxlen
augmentStartCycle = 130

augmentEndCycle = 362

maxlen=200

movingAverage = None

padded_train, train_ruls, maxlen = getPaddedSequence(scaled_train, maxlen=maxlen, augmentStartCycle=augmentStartCycle, augmentEndCycle=augmentEndCycle, movingAverage=movingAverage)

padded_test, test_ruls, maxlen = getPaddedSequence(scaled_test, maxlen=maxlen, movingAverage=movingAverage)
sns.distplot(train_ruls)
def plotDataForIndex(data, theIndex):

    plt.figure(figsize=(30, 30))

    for i in range(5, 26):

        plt.subplot(7, 3, i-4)

        values = data[theIndex][:, i]

        plt.plot(range(len(values)) ,values)

        plt.title('sensor'+str(i-4))

        plt.tight_layout()
plotDataForIndex(padded_train, 450)
# selected_sensors = [7, 8, 9, 12, 16, 17, 20]

selected_sensors = [2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]

# selected_sensors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

selected_sensors_indices = [x-1 for x in selected_sensors] # -1 because the index starts from 1
# X_train = padded_train[:, :, 5:26]

X_train = padded_train[:, :, 5:26][:, :, selected_sensors_indices]
# X_test = padded_test[:, :, 5:26]

X_test = padded_test[:, :, 5:26][:, :, selected_sensors_indices]
y_train = np.array(train_ruls).reshape(-1,1)

y_test = np.array(test_ruls).reshape(-1,1)
numOfSensors = len(X_train[0][0])
import codecs, json

def exportNPArrayToJSON(a, fileName):

    b = a.tolist() # nested lists with same data, indices

    json.dump(b, codecs.open(fileName, 'w', encoding='utf-8')) ### this saves the array in .json format
test_FD = 'test_FD00' + str(setNumber) + ".json"

train_FD = 'train_FD00' + str(setNumber) + ".json"

test_RUL_FD = 'test_RUL_FD00' + str(setNumber) + ".json"

train_RUL_FD = 'train_RUL_FD00' + str(setNumber) + ".json"

# exportNPArrayToJSON(X_train, train_FD)

exportNPArrayToJSON(X_test, test_FD)

# exportNPArrayToJSON(y_train, train_RUL_FD)

exportNPArrayToJSON(y_test, test_RUL_FD)
from IPython.display import FileLink
FileLink(test_FD)
FileLink(train_FD)
FileLink(test_RUL_FD)
FileLink(train_RUL_FD)
from keras import regularizers

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import Activation

from keras.layers import Dropout

from keras.layers import Flatten



# from keras import backend as K

# K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=36, inter_op_parallelism_threads=36)))





def createModel(l1Nodes, l2Nodes, d1Nodes, d2Nodes, inputShape):

    # input layer

    lstm1 = LSTM(l1Nodes, input_shape=inputShape, return_sequences=True, kernel_regularizer=regularizers.l2(0.1))

    do1 = Dropout(0.2)

    

    lstm2 = LSTM(l2Nodes, return_sequences=True, kernel_regularizer=regularizers.l2(0.1))

    do2 = Dropout(0.2)

    

    flatten = Flatten()

    

    dense1 = Dense(d1Nodes, activation='relu', kernel_regularizer=regularizers.l2(0.1))

    do3 = Dropout(0.2)

    

    dense2 = Dense(d2Nodes, activation='relu', kernel_regularizer=regularizers.l2(0.1))

    do4 = Dropout(0.2)

    

    # output layer

    outL = Dense(1, activation='relu', kernel_regularizer=regularizers.l2(0.1))

    # combine the layers

#     layers = [lstm1, do1, lstm2, do2, dense1, do3, dense2, do4, outL]

    layers = [lstm1, lstm2, do2, flatten,  dense1, dense2, outL]

    # create the model

    model = Sequential(layers)

    model.compile(optimizer='adam', loss='mse')

    return model
model = createModel(64, 64, 64, 8, (maxlen, numOfSensors))
model.summary()
from keras.models import Sequential

from keras.layers import Convolution1D

from keras.layers import MaxPooling1D

from keras.layers import Flatten

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import Activation

from keras.layers import Dropout
def createCNNLSTMModel(inputShape):

    cv1 = Convolution1D(input_shape=inputShape, filters=18, kernel_size=2, strides=1, padding='same', activation='relu', name='cv1')

    mp1 = MaxPooling1D(pool_size=2, strides=2, padding='same', name = 'mp1')

    

    cv2 = Convolution1D(filters=36, kernel_size=2, strides=1, padding='same', activation='relu', name='cv2')

    mp2 = MaxPooling1D(pool_size=2, strides=2, padding='same', name= 'mp2')

    

    cv3 = Convolution1D(filters=72, kernel_size=2, strides=1, padding='same', activation='relu', name='cv3')

    mp3 = MaxPooling1D(pool_size=2, strides=2, padding='same', name= 'mp3')

    

    d4 = Dense(inputShape[0]*inputShape[1], activation='relu')

    do4 = Dropout(0.2)

    

    lstm5 = LSTM(inputShape[1]*3, return_sequences=True)

    do5 = Dropout(0.2)

    

    lstm6 = LSTM(inputShape[1]*3)

    do6 = Dropout(0.2)

    

    d7 = Dense(50, activation='relu')

    do7 = Dropout(0.2)

    

    dout = Dense(1)

    

    model = Sequential([cv1, mp1, cv2, mp2, cv3, mp3, d4, do4, lstm5, do5, lstm6, do6, d7, do7, dout])

    model.compile(optimizer='rmsprop', loss='mse')

    return model
# model = createCNNLSTMModel((maxlen, numOfSensors))
# model.summary()
# from keras.models import Sequential

# from keras.layers import Convolution1D

# from keras.layers import MaxPooling1D

# from keras.layers import AveragePooling1D

# from keras.layers import Flatten

# from keras.layers import LSTM

# from keras.layers import Dense

# from keras.layers import Activation

# from keras.layers import Dropout
# def createAVGLSTMModel(inputShape):

#     a1 = AveragePooling1D(pool_size=5,stride=1, padding='same', input_shape=inputShape)  

# #     d4 = Dense(inputShape[0]*inputShape[1], activation='relu')

# #     do4 = Dropout(0.5)

    

#     lstm5 = LSTM(64, return_sequences=True)

#     do5 = Dropout(0.5)

    

#     lstm6 = LSTM(64)

#     do6 = Dropout(0.5)

    

#     d7 = Dense(8, activation='relu')

#     do7 = Dropout(0.5)

    

#     d8 = Dense(8, activation='relu')

#     do8 = Dropout(0.5)



    

#     dout = Dense(1)

    

#     model = Sequential([a1, lstm5, do5, lstm6, do6, d7, do7, d8, do8, dout])

#     model.compile(optimizer='adam', loss='mse')

#     return model
# model = createAVGLSTMModel((maxlen, numOfSensors))

# model.summary()
from keras.callbacks import ModelCheckpoint

from keras.callbacks import EarlyStopping

from keras.models import load_model

# ten fold

from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=3, shuffle=True)

from keras.models import load_model

msescores = []

counter= 0

for trainIdx, testIdx in kfold.split(X_train, y_train):

    counter = counter + 1

    # create callbacks

    model_path = 'best_model_set'+str(setNumber)+'fold'+str(counter)+'.h5'

    mc = ModelCheckpoint(model_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1)

    # create model

    # model = createModel(64, 64, 8, 8, (maxlen, numOfSensors))

    model = createCNNLSTMModel((maxlen, numOfSensors))

    model.fit(X_train[trainIdx], y_train[trainIdx], validation_data=(X_train[testIdx], y_train[testIdx]), batch_size=32, epochs=4, callbacks=[mc, es])

    # Done load the best model of this fold

    saved_model = load_model(model_path)

    msescores.append({'path': model_path, 'mse': saved_model.evaluate(X_train[testIdx], y_train[testIdx])})
msescores


for md in msescores:

    saved_model = load_model(md['path'])

    print(saved_model.evaluate(X_test, y_test))
predicted = saved_model.predict(X_test)
plt.figure(figsize=(50, 10))

plt.plot(range(len(predicted)), predicted, '-x', label='predicted')

plt.plot(range(len(y_test)), y_test, '-o', label='actual')

plt.legend()
from keras.models import Model

def getVizModel(model):

    output_layers = [l.output for l in model.layers]

    viz_model = Model(saved_model.input, output_layers)

    return viz_model
viz_model = getVizModel(saved_model)
layer_outputs = viz_model.predict(X_train)
layer_outputs[0].shape
import math

def plotLayerData(layer_data, mcIndex):

    mcData = layer_data[mcIndex]

    plt.figure(figsize=(30, 30))

    nCols = 2

    nRows = math.ceil(len(mcData[0])/nCols)

    for i in range(len(mcData[0])):

        plt.subplot(nRows, nCols, i+1)

        plt.plot(range(len(mcData[:, i])), mcData[:, i])

        plt.tight_layout()
plotLayerData(layer_outputs[1], 100)
import seaborn as sns

def plotLayerHeatmap(layer_data, mcIndex):

    mcData = layer_data[mcIndex]

    plt.figure(figsize=(30, 10))

    sns.heatmap(mcData.transpose())
plotSensorDataOfId(train, 100)
plotLayerHeatmap(layer_outputs[1], 100)