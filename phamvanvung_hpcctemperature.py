import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
import json

with open("../input/service0206-0806/service1906_1506.json") as of:

    data = json.load(of)
computes = [c for c in data.keys() if c!="timespan"]

variables = [v for v in data[computes[0]] if v!='index' and v!='arrJob_scheduling']
#Check empty array

def getEmptyArr(data, c):

    cObj = data[c]

    cDf = pd.DataFrame()

    cDf['compute'] = [c for _ in data['timespan']]

    cDf['timespan'] = data['timespan']

    for v in variables:

        vArr = np.array(cObj[v])

        if len(vArr)==0:

            print('c=', c)

            print('v=', v)

for c in computes:

    getEmptyArr(data, c)
def addTarget(cDf, predictedVar, predictedStep):

    cDf[target] = cDf[predictedVar].shift(-predictedStep)

    cDf.dropna(inplace=True)
def getComputeDf(data, c, predictedVar, predictedStep):

    cObj = data[c]

    cDf = pd.DataFrame()

    cDf['compute'] = [c for _ in data['timespan']]

    cDf['timespan'] = data['timespan']

    for v in variables:

        vArr = np.array(cObj[v])

        if len(vArr)==0:

            return None

        else:

            for i in range(len(vArr[0])):

                cDf[v+str(i)] = vArr[:, i]

    cDf['timespan'] = pd.to_datetime(cDf['timespan'])

    addTarget(cDf, predictedVar, predictedStep)

    return cDf
predictedVar = 'arrTemperature0'

target = predictedVar + "_target"

predictedSteps = 4

df = pd.concat([x for x in [getComputeDf(data, c, predictedVar, predictedSteps) for c in computes] if type(x)!="NoneType"])
df = df.reset_index().drop('index', axis=1)
features = [x for x in df.columns if x not in ['compute', 'timespan', 'arrTemperature0_target']]
features
from matplotlib import pyplot as plt

import seaborn as sns
def plotAttrDataOfId(data, compute, features):

    plt.figure(figsize=(30, 20))

    for i, v in enumerate(features):

        plt.subplot(10, 3, i+1)

        cDf = df[df['compute']==compute]

        plt.plot(cDf['timespan'], cDf[v])

        plt.title(v)

        plt.tight_layout()
for x in np.random.randint(0, len(computes), 3):

    plotAttrDataOfId(df, computes[x], features)
def plotDataDistribution(data, features):

    plt.figure(figsize=(30, 10))

    for i, v in enumerate(features):

        plt.subplot(3, 10, i+1)

        sns.distplot(list(data[v].values))

        plt.title(v)

    plt.tight_layout()
plotDataDistribution(df, features)
X_dfs = []

y = []

numberOfSequences = 400

sequenceSteps = 5

# generate training data.

for compute in computes:

    cDf = df[df['compute']==compute]

    if(len(cDf) > sequenceSteps):

        randSteps = np.random.randint(0, len(cDf)-sequenceSteps, numberOfSequences)

        for randStep in randSteps:

            X_dfs.append(cDf.iloc[randStep:randStep+sequenceSteps])

            y.append(X_dfs[-1][target].values[-1])
from sklearn.model_selection import train_test_split

X_train_dfs, X_test_dfs, y_train, y_test = train_test_split(X_dfs, y, test_size=0.33)
# combine the training data to create a scaler

train_dfs = pd.concat(X_train_dfs)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(train_dfs[features].values)
X_train = np.array([scaler.transform(item[features].values) for item in X_train_dfs])

X_test = np.array([scaler.transform(item[features].values) for item in X_test_dfs])
y_train = np.array(y_train)

y_test = np.array(y_test)
sns.distplot(y_train)
sns.distplot(y_test)
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

    

    dense1 = Dense(d1Nodes, activation='relu')

    do3 = Dropout(0.2)

    

    dense2 = Dense(d2Nodes, activation='relu')

    do4 = Dropout(0.2)

    

    # output layer

    outL = Dense(1, activation='relu')

    # combine the layers

#     layers = [lstm1, do1, lstm2, do2, dense1, do3, dense2, do4, outL]

    layers = [lstm1, lstm2, flatten,  dense1, dense2, outL]

    # create the model

    model = Sequential(layers)

    model.compile(optimizer='adam', loss='mse')

    return model
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

    model_path = 'best_model_fold'+str(counter)+'.h5'

    mc = ModelCheckpoint(model_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1)

    # create model

    model = createModel(64, 64, 8, 8, (X_train.shape[1], X_train.shape[2]))

    model.fit(X_train[trainIdx], y_train[trainIdx], validation_data=(X_train[testIdx], y_train[testIdx]), batch_size=32, epochs=40, callbacks=[mc, es])

    # Done load the best model of this fold

    saved_model = load_model(model_path)

    msescores.append({'path': model_path, 'mse': saved_model.evaluate(X_train[testIdx], y_train[testIdx])})
msescores
for md in msescores:

    saved_model = load_model(md['path'])

    print(saved_model.evaluate(X_test, y_test))
best_model = load_model(msescores[np.argmin([sc['mse'] for sc in msescores])]['path'])
predicted = saved_model.predict(X_test)
baseline = np.array([df[predictedVar].values[-1] for df in X_test_dfs])
plt.figure(figsize=(50, 10))

plt.plot(range(50), predicted[:50], 'x', label='predicted')

plt.plot(range(50), baseline[:50], 'v', label='baseline')

plt.plot(range(50), y_test[:50], 'o', label='actual')

plt.legend()
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predicted)

msebaseline = mean_squared_error(y_test, baseline)
print('mse=', mse)

print('msebaseline=', msebaseline)