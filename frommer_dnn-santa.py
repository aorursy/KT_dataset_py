import numpy as np

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, BatchNormalization

from keras.callbacks import EarlyStopping

from keras import regularizers

import time

from datetime import datetime

from keras.optimizers import Adam

from sklearn.utils import class_weight

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score
traindata = pd.read_csv('../input/train.csv')

traindata = traindata.sample(frac=1).reset_index(drop=True)

train_lable = traindata['target']

train_input = traindata.drop(['target', 'ID_code'], axis=1)

print("loading done")
def split(X, Y, splitFrac):

    n = X.shape[0]

    nf = (int)((float)(X.shape[0]) * splitFrac)

    xsplits = np.split(X, [n-2*nf, n-nf, n])

    ysplits = np.split(Y, [n-2*nf, n-nf, n])

    return xsplits[0], ysplits[0], xsplits[1], ysplits[1], xsplits[2], ysplits[2]
def normalize(df, withGiven=False, meanIn = 0, varIn = 0):

    orig_columns = df.columns

    mean = meanIn if withGiven else df.mean(axis=0)

    var = varIn if withGiven else df.var(axis=0)

    newDf = pd.DataFrame()

    for i in range(len(orig_columns)):

        f = orig_columns[i]

        newDf[f] = df[f]#(df[f] - mean[i])/var[i]

        newDf['sq_'+f] = np.power(newDf[f], 2)

        newDf['sqrt_'+f] = np.power(np.abs(newDf[f]), 0.5)

        newDf['qube_'+f] = np.power(np.abs(newDf[f]), 3)

    newDf.iloc[1:20,0:25].describe

    return newDf, mean, var



def calcWeights(Y):

    nCount = Y[Y == 0].shape[0]

    pCount = Y[Y == 1].shape[0]

    if nCount > pCount:

        return { 0: 1.0, 1: nCount/pCount}

    else:

        return { 0: pCount/nCount, 1: 1.0}
X, mean, var = normalize(train_input)

Y = train_lable



Xtrain, Ytrain, Xdev, Ydev,Xtest, Ytest = split(X, Y, 0.01)



epsilon = 0.0001

weights = class_weight.compute_class_weight('balanced',np.unique(Y), Y)

weights = dict(enumerate(weights))

def fitModel(dims, Xt, Yt, Xd, Yd, epochs, batchsize):

    model = Sequential()

    prevn = Xt.shape[1]

    for (n, act, dr, reg) in dims:

        kr2 = None if reg < epsilon else regularizers.l2(reg)

        model.add(Dense(n, activation=act, input_dim=prevn, kernel_regularizer=kr2)) #kernel_initializer='random_uniform'

        if dr > epsilon:

            model.add(Dropout(dr))

        model.add(BatchNormalization())

        nprev = n



    #opt = Adam(lr=0.001, decay=0.0001)

    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    

    start = time.time()

    hist = model.fit(Xt, Yt, epochs=epochs, batch_size=batchsize, verbose=1, shuffle=True, validation_data=(Xd,Yd), callbacks = [es], class_weight=weights)

    end = time.time()

    val_acc = hist.history['val_acc'][-1] #last element

    print("(",epochs,batchsize,")", "took", (int)((end-start)*1000/1000), "sec, acc", val_acc, "and loss acc hist ", hist.history['acc'])

    return (model, hist, (end-start)*1000)
epochs = 200

batchsize = 300

params = [[(1025,  "relu",    0.0, 0.00),        

           (256,   "relu",    0.0, 0.00),

           (16,    "relu",    0.0, 0.00),

           (1,     "sigmoid", 0.0, 0.00)]]

hist = []

for dims in params:        

    hist.append(fitModel(dims, Xtrain, Ytrain, Xdev, Ydev, epochs, batchsize))

print(hist)
def score(model, X, Y):

    devRaw = model.predict(X, batch_size=300, steps=None)

    devOut = (devRaw > 0.5)*1

    devOut = np.squeeze(np.asarray(devOut))

    tn, fp, fn, tp = confusion_matrix(Y, devOut).ravel()

    pOrig = sum(Y)

    nOrig = Y.shape[0] - pOrig

    pTarg = sum(devOut)

    nTarg = devOut.shape[0] - pTarg

    f1 = f1_score(Y, devOut)

    return((fp, fn, tp, tn, pOrig, nOrig, pTarg, nTarg, f1))
for i in range(len(hist)):

    model = hist[i][0] 

    res1 = score(model, Xtrain, Ytrain)

    res2 = score(model, Xdev, Ydev)

    res3 = score(model, Xtest, Ytest)

    print(i, "train", res1)

    print(i, "dev", res2)

    print(i, "test", res3)

import matplotlib.pyplot as plt

clrs1=['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'w-']

clrs2=['b--','g--','r--','c--','m--','y--','k--','w--']

bestModel = None

bestAcc = 0

for i in range(len(hist)):

    plt.plot(hist[i][1].history['acc'], clrs1[i])

    plt.plot(hist[i][1].history['val_acc'], clrs1[i]+'-')

    if bestAcc < hist[i][2]:

        bestModel = hist[i][0] 

        bestAcc = hist[i][2]
print("ready to test")

test_input = pd.read_csv('../input/test.csv')

test_input = test_input.drop(['ID_code'], axis=1)

Xsub = normalize(test_input, True, mean, var)[0]

targets = (bestModel.predict(Xsub, batch_size=300) > 0.5)*1

print("saving output")

toSubmit = pd.read_csv('../input/sample_submission.csv')

toSubmit['target'] = targets

filename = "sub-{:%y%m%d%H%M}.csv".format(datetime.now())

toSubmit.to_csv(filename, index=False)

print("saved", filename)