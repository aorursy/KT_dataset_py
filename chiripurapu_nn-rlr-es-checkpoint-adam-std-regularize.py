import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

print(train_data.target.value_counts())
print(train_data.shape)
train_data.isnull().any().any()
train_X = train_data.drop(['ID_code', 'target'], axis = 1)

train_Y = train_data['target']

print(train_X.shape)

print(train_Y.shape)
test_data = pd.read_csv('../input/test.csv')

print(test_data.shape)

test_data.isnull().any().any()
test_data_X = test_data.drop(['ID_code'], axis = 1)

print(test_data_X.shape)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

train_X = sc.fit_transform(train_X)

test_data_X = sc.fit_transform(test_data_X)
#from sklearn.model_selection import train_test_split

#X_train, X_test, Y_train, Y_test = train_test_split(train_X, train_Y, test_size=0.20, random_state=111)
# import the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense, Dropout , BatchNormalization

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

from keras.callbacks import ReduceLROnPlateau

from keras.utils import np_utils

from keras.optimizers import SGD

from keras import regularizers

from keras.constraints import max_norm

# Initialize the NN 



def CreateNetwork():

    model = Sequential()

    model.add(Dense(256, input_dim=X_train.shape[1] , activation='relu', 

                kernel_regularizer=regularizers.l2(0.01)))

    model.add(Dropout(0.5))

    model.add(BatchNormalization())

    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

    model.add(Dropout(0.5))

    model.add(BatchNormalization())

    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

    model.add(Dropout(0.5))

    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

    model.add(Dropout(0.5))

    model.add(BatchNormalization())

    model.add(Dense(1, activation='sigmoid'))

    #opt = SGD(lr=0.0001, momentum=0.9)

    #opt = Adam(lr=0.0001, momentum=0.9)

    #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    #model.compile(loss='binary_crossentropy', optimizer=Adam(lr = 0.001, decay=0.001/50), metrics=['accuracy'])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model
# fit model



#from sklearn.model_selection import train_test_split

#X_train, X_val, Y_train, Y_val = train_test_split(train_X, train_Y, test_size=0.20, random_state=111)





#model = CreateNetwork()

checkpoint = ModelCheckpoint('rc_model.h5', monitor='val_acc', verbose=1, 

                             save_best_only=True, mode='max', save_weights_only = True)





reduceLROnPlat = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=4, 

                                   verbose=1, mode='max', min_delta=0.0001)

# early stopping

earlyStoping = EarlyStopping(monitor='val_loss', 

                                  patience=9, mode='max', verbose=1)



callbacks_list = [checkpoint,earlyStoping,reduceLROnPlat]



#history = model.fit(X_train, Y_train, batch_size = 1024, epochs=150, validation_split=0.20, 

            #validation_data=(X_val, Y_val), callbacks=callbacks_list)





from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score



model = CreateNetwork()   





splits = list(StratifiedKFold(n_splits=5, shuffle=True).split(train_X, train_Y))



y_test_pred_log = np.zeros(len(train_X))

y_train_pred_log = np.zeros(len(train_X))

print(y_test_pred_log.shape)

print(y_train_pred_log.shape)

score = []



for i, (train_idx, test_idx) in enumerate(splits):  



    print("FOLD %s" % (i + 1))

    X_train_fold, X_val_fold  = train_X[train_idx], train_X[test_idx]

    Y_train_fold, Y_val_fold = train_Y[train_idx], train_Y[test_idx]

     

    

    

    

    history = model.fit(X_train_fold, Y_train_fold, batch_size = 32, epochs=100,

                        validation_data=(X_val_fold, Y_val_fold), callbacks=callbacks_list)

    

    model.load_weights('rc_model.h5')

    

    prediction = model.predict(X_val_fold,

                               batch_size=512,

                               verbose=1)

    # print(prediction.shape)

    # prediction = np.sum(prediction, axis=1)/2

    score.append(roc_auc_score(Y_val_fold, prediction))

    

    prediction = model.predict(test_data_X,

                               batch_size=512,

                               verbose=1)

    # y_test_pred_log += np.sum(prediction, axis=1)/2

    y_test_pred_log += prediction.reshape(prediction.shape[0])

    

    prediction = model.predict(train_X,

                               batch_size=512,

                               verbose=1)

    # y_train_pred_log += np.sum(prediction, axis=1)/2

    y_train_pred_log += prediction.reshape(prediction.shape[0])



    

    del X_train_fold, Y_train_fold, X_val_fold, Y_val_fold



print("OOF score: ", roc_auc_score(train_Y, y_train_pred_log/5))

print("average {} folds score: ".format(5), np.sum(score)/5)
#_, train_acc = model.evaluate(X_train, Y_train, verbose=0)

#print(train_acc)
#_, test_acc = model.evaluate(X_test, Y_test, verbose=0)

#print(test_acc)
# plot accuracy learning curves

plt.subplot(212)

plt.title('Accuracy', pad=-40)

plt.plot(history.history['acc'], label='train')

plt.plot(history.history['val_acc'], label='test')

plt.legend()

plt.show()
# plot loss learning curves

plt.subplot(211)

plt.title('Cross-Entropy Loss', pad=-40)

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show()
#preds = model.predict(test_data_X)
submission = pd.DataFrame({"ID_code" : test_data['ID_code'].values,

                           "target" : y_test_pred_log/5})

submission.to_csv('submission.csv', index = False, header = True)

display(submission.head(15))

display(submission.tail(15))


