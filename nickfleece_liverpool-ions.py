#imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import *

import seaborn as sns

import scipy

from sklearn.metrics import f1_score

import statistics



#This is because the data is split into 500000 rows

BATCH_LENGTH = 500000
train = pd.read_csv("../input/liverpool-ion-switching/train.csv")

test = pd.read_csv("../input/liverpool-ion-switching/test.csv")



train.head()
#Define the function to graph the signals for a set of data :)

def graphSignalChannelData(data, includeSignal=True):

    for i in range(int(len(data) / BATCH_LENGTH)):

        start = BATCH_LENGTH * i

        end = BATCH_LENGTH + (i * BATCH_LENGTH) - 1

        print(f"{i} : {start}, {end}")

        y_signal = data["signal"][start:end]

        x = range(start,end)

        plt.figure(figsize=(30,5))

        plt.plot(x, y_signal, color='g', label="Signal")

        if includeSignal:

            y_channels = data["open_channels"][start:end]

            plt.plot(x, y_channels, color='b', label="Channels")

        plt.legend()

        plt.show()
#Now graph our training data :)

graphSignalChannelData(train)
#Lets look at some distributions

def distGraphData(data):

    for i in range(10):

        start = BATCH_LENGTH * i

        end = BATCH_LENGTH + (i * BATCH_LENGTH)

        print(f"{i} : {start}, {end}")

        signals = data["signal"][start:end]

        plt.figure(figsize=(30,5))

        sns.distplot(signals)

        plt.show()
distGraphData(train)
#This is going to be our un-drifted training data

train2 = train.copy()
#We have linear drift from 500000 --> 600000 in batch 1 (500000:1000000)

plt.figure(figsize=(30,5))

plt.plot(range(500000,1000000), train['signal'][500000:1000000])

plt.title("1: No Change")

plt.show()



a=500000; b=600000 # CLEAN TRAIN BATCH 2

train2.loc[train.index[a:b],'signal'] = train2.signal[a:b].values - 3*(train2.time.values[a:b] - 50)/10.

plt.figure(figsize=(30,5))

plt.plot(range(500000,1000000), train2['signal'][500000:1000000])

plt.title("1: Undrifted")

plt.show()
def f(x,low,high,mid): return -((-low+high)/625)*(x-mid)**2+high -low



# CLEAN TRAIN BATCH 6

batch = 7; a = 500000*(batch-1); b = 500000*batch

train2.loc[train2.index[a:b],'signal'] = train.signal.values[a:b] - f(train.time[a:b].values,-1.817,3.186,325)

# CLEAN TRAIN BATCH 7

batch = 8; a = 500000*(batch-1); b = 500000*batch

train2.loc[train2.index[a:b],'signal'] = train.signal.values[a:b] - f(train.time[a:b].values,-0.094,4.936,375)

# CLEAN TRAIN BATCH 8

batch = 9; a = 500000*(batch-1); b = 500000*batch

train2.loc[train2.index[a:b],'signal'] = train.signal.values[a:b] - f(train.time[a:b].values,1.715,6.689,425)

# CLEAN TRAIN BATCH 9

batch = 10; a = 500000*(batch-1); b = 500000*batch

train2.loc[train2.index[a:b],'signal'] = train.signal.values[a:b] - f(train.time[a:b].values,3.361,8.45,475)



plt.figure(figsize=(30,5))

plt.plot(range(0, 5000), train['signal'][::1000], color='r')

plt.plot(range(0, 5000), train2['signal'][::1000], color='g')

plt.show()
patterns = {

    "v_low":[0,1],

    "low":[2,6],

    "med":[3,7],

    "high":[5,8],

    "v_high":[4,9]

}



models = {}
RANGE = 200

OFFSET = 1500000

WINDOW = 3

WINDOW_OFFSET = 1



signals = []

for index, row in train2.iloc[OFFSET:].iterrows():

    if row['open_channels'] != 0:

#         print(row['signal'])

        indices = range(index - RANGE, index + RANGE)

        signal = []

        signal_2 = []

        channels = []

        for i in indices:

            row_index = train2.iloc[i]

            signal.append(row_index.signal)

            signal_2.append(row_index.signal ** 10)

            channels.append(row_index.open_channels)

        

        data_df = pd.DataFrame(data=signal,columns=["signal"])

        

#         f = scipy.signal.hilbert(signal)

        

        plt.figure(figsize=(30,10))

        

        plt.plot(indices, signal, color='r')

        plt.plot(indices, channels, color='g')

        plt.plot(indices, signal_2, color='y')

#         plt.plot(indices, f, color='b')

        

        plt.show()

        break
train2.head()
batches = []

for i in range(10):

    start = BATCH_LENGTH * i

    end = BATCH_LENGTH + (i * BATCH_LENGTH)

    for j in range(start,end):

        batches.append(i)

train2['batch'] = batches
train2.groupby('batch')[['signal','open_channels']].agg(['min', 'max', 'median'])
#Some constants

WINDOW = 3

MIN_MAX_WINDOW = 2

WINDOW_OFFSET = 1

SHIFT_WINDOW = 2
def repeatedFeatures(data, data_df, exponent):

    

    change = pd.Series(data).diff().to_numpy()

    change[0] = 0

    data_df[f"change_exp_{exponent}"] = change

    

    pct_change = pd.Series(data).pct_change().to_numpy()

    pct_change[0] = 0

    for i in range(len(pct_change)):

        if np.isinf(pct_change[i]):

            pct_change[i] = change[i]

    data_df[f"pct_change_exp_{exponent}"] = pct_change

    

    rolling_min = data_df.rolling(MIN_MAX_WINDOW).min()[f"signal_exp_{exponent}"].to_numpy()

    rolling_min[:(WINDOW-1)] = [min(data_df[f"signal_exp_{exponent}"])] * (MIN_MAX_WINDOW - 1)

    data_df[f"rolling_min_exp_{exponent}"] = rolling_min

    

    rolling_max = data_df.rolling(MIN_MAX_WINDOW).max()[f"signal_exp_{exponent}"].to_numpy()

    rolling_max[:(WINDOW-1)] = [max(data_df[f"signal_exp_{exponent}"])] * (MIN_MAX_WINDOW - 1)

    data_df[f"rolling_max_exp_{exponent}"] = rolling_max

    

    #Shift the signal by 1 -> WINDOW and add the features

    for i in range(1, SHIFT_WINDOW + 1):

        shift = data_df.signal.shift(i).fillna(0).to_numpy()

        data_df[f"shift_{i}_exp_{exponent}"] = shift



        neg_shift = data_df.signal.shift(-i).fillna(0).to_numpy()

        data_df[f"neg_shift_{i}_exp_{exponent}"] = neg_shift

        

    return data_df
def generateFeatures(data):

    data_df = pd.DataFrame(data=data,columns=["signal"])

    

    rolling_mean = data_df.rolling(WINDOW).mean().signal.to_numpy()

    rolling_mean = np.concatenate(([statistics.mean(signal[:(WINDOW-WINDOW_OFFSET)])] * (WINDOW-WINDOW_OFFSET), rolling_mean[(WINDOW):], [statistics.mean(signal[(WINDOW_OFFSET):])] * (WINDOW_OFFSET)))

    data_df["rolling_mean"] = rolling_mean

    

    rolling_median = data_df.rolling(WINDOW).median().signal.to_numpy()

    rolling_median = np.concatenate(([statistics.median(signal[:(WINDOW-WINDOW_OFFSET)])] * (WINDOW-WINDOW_OFFSET), rolling_median[(WINDOW):], [statistics.median(signal[(WINDOW_OFFSET):])] * (WINDOW_OFFSET)))

    data_df["rolling_median"] = rolling_median

    

    rolling_min = data_df.rolling(MIN_MAX_WINDOW).min().signal.to_numpy()

    rolling_min[:(WINDOW-1)] = [min(data)] * (MIN_MAX_WINDOW - 1)

    data_df["rolling_min"] = rolling_min

    

    rolling_max = data_df.rolling(MIN_MAX_WINDOW).max().signal.to_numpy()

    rolling_max[:(WINDOW-1)] = [max(data)] * (MIN_MAX_WINDOW - 1)

    data_df["rolling_max"] = rolling_max

    

    #This seems to lower the score :(

#     std = data_df.signal.std()

#     mean = data_df.signal.mean()

#     normalized_signal = ((data_df.signal - mean) / std).to_numpy()

#     data_df["normalized_signal"] = normalized_signal

    

    exponents = [2]

    for exp in exponents:

        signal_exp = []

        for datum in data:

            signal_exp.append(datum ** exp)

        data_df[f"signal_exp_{exp}"] = signal_exp

        data_df = repeatedFeatures(signal_exp, data_df, exp)

    

    return data_df
for i in patterns.keys():

    print(f"STARTING {i} CLASSIFIER")



    batches = patterns[i]

    batch0 = train2.loc[train2["batch"] == batches[0]]

    batch1 = train2.loc[train2["batch"] == batches[1]]

    batch_df = pd.concat([batch0, batch1])

    

    num_channels = len(batch_df.open_channels.unique())

    

    channels = batch_df["open_channels"]



#     clf = tree.DecisionTreeClassifier()

#     clf = ensemble.RandomForestClassifier()

    clf = ensemble.GradientBoostingClassifier(verbose=1, n_estimators=150, learning_rate=0.2)

#     clf = svm.SVC()

#     clf = neural_network.MLPClassifier()

    

    print(f"GENERATING FEATURES...")

    data_points = generateFeatures(batch_df["signal"])

    print(f"FITTING MODEL...")

    clf.fit(data_points, channels)

    print(f"DONE FITTING!")

    models[i] = clf

    print(f"Score: {clf.score(data_points, channels)}")

    print("\n")

    print("Feature ranking:")

    feature_importances = clf.feature_importances_

    scores = []

    for score, col in zip(feature_importances, data_points.columns):

        if score != 0.0:

            scores.append([col, score])

    scores.sort(reverse=True,key=lambda x: x[1])

    for score in scores:

        print(f"{score[0]} : {score[1]}")

    

    print("\n\n")
for i in patterns.keys():

    print(f"Checking classifier results for {i}")

    

    batches = patterns[i]

    batch0 = train2.loc[train2["batch"] == batches[0]]

    batch1 = train2.loc[train2["batch"] == batches[1]]

    batch_df = pd.concat([batch0, batch1])

    

    signal = batch_df["signal"].to_numpy()

    open_channels = batch_df["open_channels"].to_numpy()

    

    clf = models[i]

    data_points = generateFeatures(signal)

    predictions = clf.predict(data_points)

    

    SIGNAL_RANGE = 20

    for prediction, actual, i in zip(predictions, open_channels, range(len(predictions))):

        if prediction != actual:

            signal_i = signal[i-SIGNAL_RANGE:i+SIGNAL_RANGE]

            prediction_i = predictions[i-SIGNAL_RANGE:i+SIGNAL_RANGE]

            actual_i = open_channels[i-SIGNAL_RANGE:i+SIGNAL_RANGE]

            

            plt.figure(figsize=(20,10))



            indices = range(i-SIGNAL_RANGE, i+SIGNAL_RANGE)

            plt.plot(indices, signal_i, color='r')

            plt.plot(indices, prediction_i, color='g')

            plt.plot(indices, actual_i, color='b')

            

            plt.show()

            break
graphSignalChannelData(test, False)
test2 = test.copy()
# REMOVE BATCH 0 DRIFT

start=500

a = 0; b = 100000

test2.loc[test2.index[a:b],'signal'] = test2.signal.values[a:b] - 3*(test2.time.values[a:b]-start)/10.

start=510

a = 100000; b = 200000

test2.loc[test2.index[a:b],'signal'] = test2.signal.values[a:b] - 3*(test2.time.values[a:b]-start)/10.

start=540

a = 400000; b = 500000

test2.loc[test2.index[a:b],'signal'] = test2.signal.values[a:b] - 3*(test2.time.values[a:b]-start)/10.
# REMOVE BATCH 1 DRIFT

start=560

a = 600000; b = 700000

test2.loc[test2.index[a:b],'signal'] = test2.signal.values[a:b] - 3*(test2.time.values[a:b]-start)/10.

start=570

a = 700000; b = 800000

test2.loc[test2.index[a:b],'signal'] = test2.signal.values[a:b] - 3*(test2.time.values[a:b]-start)/10.

start=580

a = 800000; b = 900000

test2.loc[test2.index[a:b],'signal'] = test2.signal.values[a:b] - 3*(test2.time.values[a:b]-start)/10.
# REMOVE BATCH 2 DRIFT

def f(x):

    return -(0.00788)*(x-625)**2+2.345 +2.58

a = 1000000; b = 1500000

test2.loc[test2.index[a:b],'signal'] = test2.signal.values[a:b] - f(test2.time[a:b].values)
res = 1000; let = ['A','B','C','D','E','F','G','H','I','J']



plt.figure(figsize=(30,5))

plt.plot(range(0, 2000000)[::1000], test['signal'][::1000], color='r')

plt.plot(range(0, 2000000)[::1000], test2['signal'][::1000], color='g')

for i in range(5): plt.plot([i*500000,i*500000],[-5,12.5],'r')

for i in range(21): plt.plot([i*100000,i*100000],[-5,12.5],'r:')

for k in range(4): plt.text(k*500000+250000,10,str(k+1),size=20)

for k in range(10): plt.text(k*100000+40000,7.5,let[k],size=16)

plt.show()
predictions = []



# SUBSAMPLE A, Model v_low

data = generateFeatures(test2["signal"].values[0:100000])

predictions = np.concatenate((predictions, models["v_low"].predict(data)))



# SUBSAMPLE B, Model med

data = generateFeatures(test2["signal"].values[100000:200000])

predictions = np.concatenate((predictions, models["med"].predict(data)))



# SUBSAMPLE C, Model high

data = generateFeatures(test2["signal"].values[200000:300000])

predictions = np.concatenate((predictions, models["high"].predict(data)))



# SUBSAMPLE D, Model v_low

data = generateFeatures(test2["signal"].values[300000:400000])

predictions = np.concatenate((predictions, models["v_low"].predict(data)))



# SUBSAMPLE E, Model med

data = generateFeatures(test2["signal"].values[400000:500000])

predictions = np.concatenate((predictions, models["med"].predict(data)))



# SUBSAMPLE F, Model v_high

data = generateFeatures(test2["signal"].values[500000:600000])

predictions = np.concatenate((predictions, models["v_high"].predict(data)))



# SUBSAMPLE G, Model high

data = generateFeatures(test2["signal"].values[600000:700000])

predictions = np.concatenate((predictions, models["high"].predict(data)))



# SUBSAMPLE H, Model v_high

data = generateFeatures(test2["signal"].values[700000:800000])

predictions = np.concatenate((predictions, models["v_high"].predict(data)))



# SUBSAMPLE I, Model v_low

data = generateFeatures(test2["signal"].values[800000:900000])

predictions = np.concatenate((predictions, models["v_low"].predict(data)))



# SUBSAMPLE J, Model med

data = generateFeatures(test2["signal"].values[900000:1000000])

predictions = np.concatenate((predictions, models["med"].predict(data)))



# BATCHES 3 AND 4, Model v_low

data = generateFeatures(test2["signal"].values[1000000:2000000])

predictions = np.concatenate((predictions, models["v_low"].predict(data)))



plt.figure(figsize=(30,5))

plt.plot(range(len(predictions))[::100], predictions[::100])

plt.show()
sub = test.copy()

sub.pop('signal')

sub["open_channels"] = [int(x) for x in predictions]

sub.to_csv('submission.csv',index=False, float_format='%.4f')