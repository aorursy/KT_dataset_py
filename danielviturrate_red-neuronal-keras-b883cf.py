# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.layers import Input, Dense, BatchNormalization, Add, GaussianNoise, Dropout

from keras.models import Model

from sklearn.metrics import roc_auc_score

from keras.layers import Wrapper

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

from keras import regularizers

import matplotlib.pyplot as plt

# Feature Scaling

from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
precisiones_globales=[]

epochs = 15

def graf_model(train_history):

    f = plt.figure(figsize=(15,10))

    ax = f.add_subplot(121)

    ax2 = f.add_subplot(122)

    # summarize history for accuracy

    ax.plot(train_history.history['binary_accuracy'])

    ax.plot(train_history.history['val_binary_accuracy'])

    ax.set_title('model accuracy')

    ax.set_ylabel('accuracy')

    ax.set_xlabel('epoch')

    ax.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    ax2.plot(train_history.history['loss'])

    ax2.plot(train_history.history['val_loss'])

    ax2.set_title('model loss')

    ax2.set_ylabel('loss')

    ax2.set_xlabel('epoch')

    ax2.legend(['train', 'test'], loc='upper left')

    plt.show()

def precision(model, registrar=False):

    y_pred = model.predict(train_dfX)

    train_auc = roc_auc_score(train_dfY, y_pred)

    y_pred = model.predict(val_dfX)

    val_auc = roc_auc_score(val_dfY, y_pred)

    print('Train AUC: ', train_auc)

    print('Vali AUC: ', val_auc)

    if registrar:

        precisiones_globales.append([train_auc,val_auc])

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print("Train shape : ",train_df.shape)

print("Test shape : ",test_df.shape)
train_dfX = train_df.drop(['ID_code', 'target'], axis=1)

train_dfY = train_df['target']

submission = test_df[['ID_code']].copy()

test_df = test_df.drop(['ID_code'], axis=1)

sc = StandardScaler()

train_dfX = sc.fit_transform(train_dfX)

test_df = sc.transform(test_df)
train_dfX,val_dfX,train_dfY, val_dfY = train_test_split(train_dfX,train_dfY , test_size=0.1, stratify=train_dfY)

print("Entrenamiento: ",train_dfX.shape)

print("Validacion : ",val_dfX.shape)
def func_model():   

    inp = Input(shape=(200,))

    x=Dense(1028, activation="relu", kernel_initializer='random_uniform', bias_initializer='zeros')(inp)

    x=Dense(1028, activation="relu", kernel_initializer='random_uniform', bias_initializer='zeros')(x) 

    x=Dense(1, activation="sigmoid", kernel_initializer='random_uniform', bias_initializer='zeros')(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['binary_accuracy'])

    return model

model = func_model()

print(model.summary())
train_history = model.fit(train_dfX, train_dfY, batch_size=512, epochs=epochs, validation_data=(val_dfX, val_dfY))

graf_model(train_history)
precision(model, True)
def func_model(arquitectura): 

    first =True

    inp = Input(shape=(200,))

    for capa in arquitectura:        

        if first:

            x=Dense(capa, activation="relu", kernel_initializer='random_uniform', bias_initializer='zeros')(inp)            

            first = False

        else:

            x=Dense(capa, activation="relu", kernel_initializer='random_uniform', bias_initializer='zeros')(x)  

    x=Dense(1, activation="sigmoid", kernel_initializer='random_uniform', bias_initializer='zeros')(x)  

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['binary_accuracy'])

    return model
arquitectura1 = [10,1000,2000]

model1 = func_model(arquitectura1)

#Para revisar la estructura del modelo, quitar el comentario de la instruccion siguiente:

print(model1.summary())

train_history_tam1 = model1.fit(train_dfX, train_dfY, batch_size=512, epochs=epochs, validation_data=(val_dfX, val_dfY), verbose=0)

graf_model(train_history_tam1)

precision(model1)
arquitectura2 = [10, 1500, 2000]

model2 = func_model(arquitectura2)

#print(model2.summary())

train_history_tam2 = model2.fit(train_dfX, train_dfY, batch_size=512, epochs=epochs, validation_data=(val_dfX, val_dfY))

graf_model(train_history_tam2)

precision(model2)
arquitecturaFinal = [10,1500,2000]

modelF = func_model(arquitecturaFinal)

print(modelF.summary())

train_history_tamF = modelF.fit(train_dfX, train_dfY, batch_size=512, epochs=epochs, validation_data=(val_dfX, val_dfY))

graf_model(train_history_tamF)

precision(modelF, True)

assert(len(precisiones_globales)==2)
def func_model_reg():   

    inp = Input(shape=(200,))

    x=Dropout(0.2)(inp)

    x=Dense(1028, activation="relu", kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01))(x)

    x=Dropout(0.5)(x)

    x=Dense(1028, activation="relu", kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=None)(x)

    x=Dropout(0.7)(x)

    x=Dense(1028, activation="relu", kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=None)(x)

    x=Dropout(0.7)(x)  

    x=Dense(1, activation="sigmoid", kernel_initializer='random_uniform', bias_initializer='zeros')(x) 

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['binary_accuracy'])

    return model
model1 = func_model_reg()

#Para revisar la estructura del modelo, quitar el comentario de la instruccion siguiente:

print(model1.summary())

train_history_tam1 = model1.fit(train_dfX, train_dfY, batch_size=512, epochs=epochs, validation_data=(val_dfX, val_dfY), verbose=0)

graf_model(train_history_tam1)

precision(model1)
modelRF = func_model_reg()

print(modelRF.summary())

train_history_regF = modelRF.fit(train_dfX, train_dfY, batch_size=512, epochs=epochs, validation_data=(val_dfX, val_dfY))

graf_model(train_history_regF)

precision(modelRF, True)

assert(len(precisiones_globales)==3)
y_test = modelRF.predict(test_df)

submission['target'] = y_test

submission.to_csv('submission.csv', index=False)