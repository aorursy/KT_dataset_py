import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import *

from sklearn.experimental import enable_hist_gradient_boosting

import lightgbm as lgb

import matplotlib.pyplot as plt

from datetime import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

train = pd.read_csv("../input/data-without-drift/train_clean.csv")

test = pd.read_csv("../input/data-without-drift/test_clean.csv")

display(train.head())



n = 500000

sets = [train.iloc[i:i + n] for i in range(0, len(train), n)]

testsets = [test.iloc[i:i + n] for i in range(0, len(test), n)]



#train.plot.scatter("time", "signal", c="open_channels", colormap="viridis", alpha=0.1)
plt.scatter(sets[7].time, sets[7].signal, c=sets[7].open_channels, cmap="viridis")

train = train.drop(sets[7].index)
fig, ax = plt.subplots(2, 1, figsize=(15, 10))

ax[0].plot(train.time, train.signal)

ax[1].plot(test.time - 500, test.signal)

letters = {"train": ["Ar", "Ar", "As", "B", "C", "D", "As", "", "D", "C"], "test": ["As", "B", "D", "Ar", "As", "C", "D", "C", "Ar", "B", "Ar", "Ar", "Ar", "Ar", "Ar", "Ar", "Ar", "Ar", "Ar", "Ar"]}



for i in range(len(sets)):

    ax[0].axvline(i * 50, color="r")

    ax[0].text(i * 50 + 20, -2, letters["train"][i], color="w")

    ax[0].set_title("train")



for i in range(len(testsets)):

    for j in range(5):

        ax[1].axvline(i * 50 + j * 10, color="r")

        ax[1].text(i * 50 + j * 10 + 5, -2.5, letters["test"][5 * i + j], color="w")

    ax[1].set_title("test")
def get_name(model):

    return str(model).split("(")[0]

print(get_name(ensemble.GradientBoostingRegressor()))
trainsets = {}

models = {}

modeltypes = [ensemble.GradientBoostingRegressor, ensemble.HistGradientBoostingRegressor]#, ensemble.RandomForestClassifier]#, neighbors.KNeighborsClassifier, tree.DecisionTreeClassifier]

f1_sum = 0.0



for i in range(len(sets)):

    pattern = letters["train"][i]

    if pattern not in trainsets.keys():

        trainsets[pattern] = sets[i]

    else:

        trainsets[pattern] = trainsets[pattern].append(sets[i])
traindata = trainsets["C"]

display(type(traindata))



average = traindata.signal.rolling(5, center=True, win_type="barthann").mean()

#average = average.fillna(traindata.signal.rolling(5, center=True, win_type="barthann").mean())

average = average.fillna(traindata.signal)

traindata["rollavg"] = average



display(traindata.rollavg.head(), traindata.rollavg.tail())

fig, ax = plt.subplots(2)

header = traindata.head(1000)



ax[0].plot(header.time, header.signal, label="signal")

ax[0].plot(header.time, header.rollavg, label="average")



ax[1].plot(header.time, header.open_channels, label="open_channels")

ax[1].plot(header.time, header.rollavg + 3.5, label="average")

plt.legend()



display(traindata.corr())
#forward = traindata.signal.shift(3).fillna(traindata.signal)

#backward = traindata.signal.shift(-3).fillna(traindata.signal)

forward = traindata.rollavg.shift(3).fillna(traindata.signal)

backward = traindata.rollavg.shift(-3).fillna(traindata.signal)

grad = np.gradient(traindata.signal)



traindata["lag"] = backward

traindata["lead"] = forward

traindata["grad"] = grad
pattern = "C"

startc = datetime.now()



print(len(traindata.signal), len(traindata.rollavg), len(traindata.lag), len(traindata.lead))

x = np.array([traindata.signal, traindata.rollavg, traindata.lag, traindata.lead, traindata.grad]).T

print(x.shape)

y = traindata.open_channels

#newmodel = ensemble.RandomForestRegressor(n_jobs=-1)

lgbtrain = lgb.Dataset(x, label=y, free_raw_data=False)

param = {"max_bin": 2048}



print("Training:", pattern)

#newmodel.fit(x, y)

newmodel = lgb.train(param, lgbtrain)

print("Predicting:", pattern)

#print(newmodel.predict(x))

pred = np.round(newmodel.predict(x))

print(metrics.f1_score(y, pred, average="micro"))

print("Finished in", datetime.now() - startc)
for trainset in trainsets.values():

    avg = trainset.signal.rolling(5, center=True).mean()

    avg = avg.fillna(trainset.signal)

    

    #forward = trainset.signal.shift(3).fillna(trainset.signal)

    #backward = trainset.signal.shift(-3).fillna(trainset.signal)

    forward = avg.shift(3).fillna(trainset.signal)

    backward = avg.shift(-3).fillna(trainset.signal)

    

    grad = np.gradient(trainset.signal)

    

    trainset["lag"] = backward

    trainset["lead"] = forward

    trainset["rollavg"] = avg

    trainset["grad"] = grad
for pattern in trainsets.keys():

    if pattern == "": continue

    mscore = 0.0

    traindata = trainsets[pattern]

    #numchannels = len(traindata.open_channels.unique())

    

    x = np.array([traindata.signal, traindata.rollavg, traindata.lag, traindata.lead, traindata.grad]).T

    y = traindata.open_channels

    lgbtrain = lgb.Dataset(x, label=y, free_raw_data=False)

    param = {"max_bin": 2048}

    #param['metric'] = 'auc'

    

    print("Training:", pattern)

    newmodel = lgb.train(param, lgbtrain)

    print("Predicting:", pattern)

    pred = np.round(newmodel.predict(x))

    print(metrics.f1_score(y, pred, average="micro"))

    models[pattern] = newmodel
n = 100000

testdata = [test.iloc[i:i+n] for i in range(0, len(test), n)]

for testd in testdata:

    avg = testd.signal.rolling(5, center=True).mean()

    avg = avg.fillna(testd.signal)

    

    #forward = testd.signal.shift(3).fillna(testd.signal)

    #backward = testd.signal.shift(-3).fillna(testd.signal)

    

    forward = avg.shift(3).fillna(testd.signal)

    backward = avg.shift(-3).fillna(testd.signal)

    

    grad = np.gradient(testd.signal)

    

    testd["lag"] = backward

    testd["lead"] = forward

    testd["rollavg"] = avg

    testd["grad"] = grad

print(testdata[0].shape)
data = np.array([testdata[0].signal, testdata[0].rollavg, testdata[0].lag, testdata[0].lead, testdata[0].grad]).T

pred = models[letters["test"][0]].predict(data)



for i in range(1, len(testdata)):

    data = np.array([testdata[i].signal, testdata[i].rollavg, testdata[i].lag, testdata[i].lead, testdata[0].grad]).T

    pred = np.append(pred, models[letters["test"][i]].predict(data))



#print(data.shape)

pred = np.round(pred).astype(int)

out = pd.DataFrame({"time": test.time, "open_channels": pred})

out.to_csv("LGB_AvgMultiFeature.csv", float_format='%0.4f', index=False)
plt.scatter(test.time, test.signal, c=out.open_channels, cmap="viridis", alpha=0.1)