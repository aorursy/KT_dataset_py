import numpy as np 

import pandas as pd 

import keras

import keras.backend as K



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold

import copy



%matplotlib inline

from matplotlib import pyplot as plt

from tqdm import tqdm_notebook as tqdm



import tensorflow.test

print(tensorflow.test.is_gpu_available())
train = pd.read_csv('../input/data_revamped/train.csv', index_col='identifier')

test = pd.read_csv('../input/data_revamped/test.csv', index_col='identifier')



#features selected based on feature importances from random forest

feats = ['f53', 'f63', 'f14', 'f15', 'f20', 'f28', 'f22', 'f4']

mfc = 0

for f1 in feats:

    mfc+=1

    train['mfc'+str(mfc)] = train[f1].apply(np.sqrt)

    test[ 'mfc'+str(mfc)] = test[ f1].apply(np.sqrt)

    for f2 in feats:

        if f1 is not f2:

            mfc+=1

            train['mfc'+str(mfc)] = train[f1] / train[f2]

            test[ 'mfc'+str(mfc)] = test[ f1] / test[ f2]         
#train, val = train_test_split(train)
y = pd.DataFrame(train.target)

#y_val = pd.DataFrame(val.target)

train.drop(['target'], 1, inplace=True)

#val.drop('target', 1, inplace=True)
train.shape
def Model():

    inp = keras.layers.Input(shape=(128,))

    X = keras.layers.BatchNormalization()(inp)

    X = keras.layers.Dense(128, activation='relu')(X)

    X = keras.layers.Dense(8, activation='relu')(X)

    #X = keras.layers.Dropout(0.3)(X)

    pred = keras.layers.Dense(1, activation='sigmoid')(X)

    model = keras.Model(inputs = inp, outputs = pred)

    

    return model
model = None

vals, sub = [], []

his = []

for i in tqdm(range(1000)):

    kf = StratifiedKFold(3, shuffle=True)

    for tr, val in kf.split(train, y):

        X_tr, X_val = train.iloc[tr], train.iloc[val]

        y_tr, y_val = y.iloc[tr], y.iloc[val]



        del model

        K.clear_session()

        

        model = Model()

        model.compile(optimizer=keras.optimizers.Adam(0.004),

                  metrics=['acc'],

                  loss='binary_crossentropy')



        chkpt = keras.callbacks.ModelCheckpoint("weights.h5", monitor='val_acc',

                                            verbose=1, save_best_only=True,

                                            save_weights_only=True, mode='auto', period=2)



        #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5,

        #                                 verbose=0, mode='auto', min_delta=0.001, cooldown=0, min_lr=0)



        model.fit(X_tr.values, y_tr.values, epochs=50, validation_data=(X_val.values, y_val.values),

                  callbacks=[chkpt], batch_size=16)



        model.load_weights('weights.h5')

        vals.append(model.evaluate(X_val.values, y_val.values))

        sub.append(model.predict(test.values))
vac = []

for val in vals: vac.append(val[1])

vac = np.array(vac)

inx = np.argsort(vac)
#selecting top classifiers

print(len(sub))

begin = 2100

indexes = inx[begin:]
print(len(indexes))
pred = pd.DataFrame(np.column_stack(np.array(sub)[indexes]).sum(1), index=test.index, columns=['target'])

pred.target = (pred.target/(len(indexes)*0.83)).apply(np.round).apply(np.int) #adjust threshold
pred.to_csv('sub.csv')
pred.describe()
pd.DataFrame(np.array(vals)[indexes]).describe()
#this gives acc of 0.86875

#ensembling 3 of these gives 0.1% increase in acc to .86975 which overfits on 20% test data, as previous solution gave .873 acc on private test data