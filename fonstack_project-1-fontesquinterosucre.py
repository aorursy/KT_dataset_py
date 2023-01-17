# Python

import time

import os

import numpy as np

# Manejo de data y gráficas (Pandas, Matplotlib)

import pandas as pd

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style

from matplotlib import cbook as cbook

# Framework (Keras)

from tensorflow.keras.layers import Input, Dense, BatchNormalization, Add, GaussianNoise, Dropout

from tensorflow.keras.models import Model

from sklearn.metrics import roc_auc_score

from tensorflow.keras.layers import Wrapper

from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

from tensorflow.keras import regularizers

from sklearn.preprocessing import StandardScaler

# Random seed

np.random.seed(1)

# Config pandas

pd.set_option('display.max_columns', None)
test_csv = pd.read_csv('../input/titanic/test.csv')

train_csv = pd.read_csv('../input/titanic/train.csv')
print(' ⮞ Train shape: ', train_csv.shape)

print(' ⮞ Observaciones en train: ', train_csv.shape[0])

print(' ⮞ Test shape: ', test_csv.shape)

print(' ⮞ Observaciones en test: ', test_csv.shape[0])
train_csv.describe()
train_dfY = train_csv.Survived

train_csv = train_csv.drop(['Survived'], axis=1)



train_test = train_csv.append(test_csv)

train_test.reset_index(inplace=True)

train_test.drop(['index'], inplace=True, axis=1)



train_test.head()
train_test.describe()
train_test['Age'] = train_test['Age'].fillna(train_test['Age'].median())

train_test.describe()
train_test['Title'] = train_test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

train_test['Title'] = train_test['Title'].replace(['Mlle', 'Ms'], 'Miss')

train_test['Title'] = train_test['Title'].replace('Mme', 'Mrs')

train_test['Title'] = train_test['Title'].replace(['Lady', 'Sir', 'Countess', 'Don', 'Dona', 'Jonkheer'], 'Royal')

# Llenamos los 'Embarked' con el valor más común

embarked_most_common = train_test['Embarked'].describe().values[2]

train_test['Embarked'] = train_test['Embarked'].fillna(embarked_most_common)

# Cambiamos valores de las variables que son strings a numéricos

sex_dictionary = { "female": 0, "male": 1 }

embarked_dictionary = {"S": 0, "C": 1, "Q": 2}

title_dictionary = { "Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Royal": 5 }

cabins_dictionary = { 'N': 0, 'C': 1, 'E': 2, 'G': 3, 'D': 4, 'A': 5, 'B': 6, 'F': 8, 'T': 9 }

train_test['Sex'] = train_test['Sex'].map(sex_dictionary)

train_test['Embarked'] = train_test['Embarked'].map(embarked_dictionary)

train_test['Title'] = train_test['Title'].map(title_dictionary)

train_test['Title'] = train_test['Title'].fillna(0)

train_test['Title'] = train_test['Title'].astype(int)

train_test['Cabin'] = train_test['Cabin'].fillna('N/A')

train_test['Cabin'] = train_test['Cabin'].apply(lambda x: x[0])

train_test['Cabin'] = train_test['Cabin'].map(cabins_dictionary)

train_test['Age'] = train_test['Age'].astype(int)

train_test['Fare'] = train_test['Fare'].fillna(train_test['Fare'].mean())

train_test['Fare'] = train_test['Fare'].astype(int)

train_test['KinfolkNumber'] = train_test['Parch'] + train_test['SibSp'] + 1

    

train_test.head()   
# Se copian los ids de los pasajeros que están en el test set

predictions = test_csv[['PassengerId']].copy()

# Se obtienen los nuevos datasets eliminando información que no se usará

train_test = train_test.drop(columns=['PassengerId', 'Name', 'Ticket'], axis=1)



train_test.head()  
# One Hot Encoding con 'Pclass'

pclass_encoding = pd.get_dummies(train_test['Pclass'], prefix='Pclass')  

train_test = pd.concat([train_test, pclass_encoding], axis=1)

# One Hot Encoding con 'Cabin'

cabin_encoding = pd.get_dummies(train_test['Cabin'], prefix='Cabin')  

train_test = pd.concat([train_test, cabin_encoding], axis=1)

# One Hot Encoding con 'Embarked'

embarked_encoding = pd.get_dummies(train_test['Embarked'], prefix='Embarked')  

train_test = pd.concat([train_test, embarked_encoding], axis=1)

# One Hot Encoding con 'Title'

title_encoding = pd.get_dummies(train_test['Title'], prefix='Title')  

train_test = pd.concat([train_test, title_encoding], axis=1)



# Se eliminan los features que ya se hicieron en One Hot Encoding

train_test.drop(['Pclass', 'Title', 'Cabin', 'Embarked'], axis=1, inplace=True)



print(' ⮞ Shape de train-test', train_test.shape)

train_test.head()
Y_train = train_dfY

X_train = train_test.iloc[:891]

X_test = train_test.iloc[891:]



print(' ⮞ Shape de train Y(output): ', Y_train.shape)

print(' ⮞ Shape de train X(input): ', X_train.shape)

print(' ⮞ Shape de test X(input): ', X_test.shape)

print(' ⮞ Cantidad de observaciones en train: ', X_train.shape[0])

print(' ⮞ Cantidad de observaciones en test: ', X_test.shape[0])

print('\n')

print(' ⮞ Cantidad de features: ', X_train.shape[1])
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
print()

X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=0.2, stratify=Y_train)

print(' ⮞ Shape de train X(input): ', X_train.shape)

print(' ⮞ Shape de dev X(input): ', X_dev.shape)
precisiones_globales=[]

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

    Y_pred = model.predict(X_train)

    train_auc = roc_auc_score(Y_train, Y_pred)

    Y_pred = model.predict(X_dev)

    val_auc = roc_auc_score(Y_dev, Y_pred)

    print('Train AUC: ', train_auc)

    print('Vali AUC: ', val_auc)

    if registrar:

        precisiones_globales.append([train_auc,val_auc])
# def model(): 

#     X = Input(shape=(27,))

#     a=Dense(12, activation = "relu", kernel_regularizer = None, kernel_initializer='glorot_normal', bias_initializer='zeros')(X)

#     a=Dense(12, activation = "relu", kernel_regularizer = None, kernel_initializer='glorot_normal', bias_initializer='zeros')(a)

#     a=Dense(10, activation = "relu", kernel_regularizer = None, kernel_initializer='glorot_normal', bias_initializer='zeros')(a)

#     a=Dense(10, activation = "relu", kernel_regularizer = None, kernel_initializer='glorot_normal', bias_initializer='zeros')(a)

#     y_hat=Dense(1, activation = "sigmoid")(a) 

#     model = Model(inputs = X, outputs = y_hat)

#     model.compile(loss = 'binary_crossentropy', optimizer = 'Adam', metrics = ['binary_accuracy'])

#     return model



# model_FQS = model()

# # print(model_FQS.summary())
# train_history_WR = model_FQS.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_dev, Y_dev), verbose=0)

# # Se imprimen gráficas

# graf_model(train_history_WR)

# precision(model_FQS)
def modelReg(): 

    X = Input(shape=(27,))

    a=Dropout(0.01)(X)

    a=Dense(20, activation = "relu", kernel_regularizer = regularizers.l2(0.01), bias_initializer='zeros')(a)

    a=Dropout(0.5)(a)

    a=Dense(20, activation = "relu", kernel_regularizer = None, bias_initializer='zeros')(a)

    a=Dropout(0.6)(a)

    a=Dense(10, activation = "relu", kernel_regularizer = None, bias_initializer='zeros')(a)

    a=Dropout(0)(a)

    y_hat=Dense(1, activation = "sigmoid")(a) 

    model = Model(inputs = X, outputs = y_hat)

    model.compile(loss = 'binary_crossentropy', optimizer = 'Adam', metrics = ['binary_accuracy'])

    return model



model_FQS_R = modelReg()

# print(model_FQS.summary())
train_history = model_FQS_R.fit(X_train, Y_train, batch_size=32, epochs=200, validation_data=(X_dev, Y_dev), verbose=0)

# Se imprimen gráficas

graf_model(train_history)

precision(model_FQS_R)
Y_test = model_FQS_R.predict(X_test)

for i in range(len(Y_test)):

    if Y_test[i] < 0.5: 

        Y_test[i] = 0

    else:

        Y_test[i] = 1        

predictions['Survived'] = Y_test.astype(int)

predictions.to_csv('submission.csv', index=False)



pred = pd.read_csv('submission.csv')

pred.head()