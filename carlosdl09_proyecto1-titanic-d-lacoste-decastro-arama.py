# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting

from keras.layers import Input, Dense, BatchNormalization, Add, GaussianNoise, Dropout

from keras.models import Model

from sklearn.metrics import roc_auc_score

from keras.layers import Wrapper

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

from keras import regularizers, initializers, optimizers



# Feature Scaling

from sklearn.preprocessing import StandardScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print("Train shape : ",train_df.shape)

print("Test shape : ",test_df.shape)
print("Train head:\n",train_df.head().to_string())

print("Test head:\n",test_df.head().to_string())

print("Train info:\n",train_df.info())

print("Test info:\n",test_df.info())
train_df.groupby(['Sex','Survived']).size().unstack().plot(kind='bar',stacked=True)

plt.show()
train_df.groupby(['Age','Survived']).size().groupby(level=0).apply(

    lambda x: 100 * x / x.sum()

).unstack().plot(kind='bar',stacked=True, figsize = (20,5))

plt.show()
train_df.groupby(['Pclass','Survived']).size().unstack().plot(kind='bar',stacked=True)

plt.show()
train_df.groupby(['Embarked','Survived']).size().unstack().plot(kind='bar',stacked=True)

plt.show()
train_df.groupby(['Embarked', 'Pclass']).size().unstack().plot(kind='bar',stacked=True)

plt.show()
tabla_completa = train_df.append(test_df, ignore_index=True)
# Obtener lista de letras de cabinas

tabla_completa["Cabin"].str[0].unique().tolist()
# Reemplazar NaN en Cabin por N (No conocido)

tabla_completa["Cabin"] = tabla_completa["Cabin"].fillna("N")

# Nueva lista de letras de cabinas

letras_cabinas = tabla_completa["Cabin"].str[0].unique().tolist()
tabla_completa["Letra_Cabina"] = tabla_completa["Cabin"].str[0]
tabla_completa.groupby(['Letra_Cabina','Survived']).size().groupby(level=0).apply(

    lambda x: 100 * x / x.sum()

).unstack().plot(kind='bar',stacked=True)

plt.show()
# Obtener lista de títulos

lista_titulos = tabla_completa["Name"].str.partition('.')[0].str.rpartition()[2].unique().tolist()

lista_titulos
dic_titulos_simples = {

    "Mr" : "Mr",

    "Mrs" : "Mrs",

    "Miss" : "Miss",

    "Master" : "Master",

    "Don" : "Nobleza",

    "Rev" : "Oficial",

    "Dr" : "Oficial",

    "Mme" : "Mrs",

    "Ms" : "Mrs",

    "Major" : "Oficial",

    "Lady" : "Nobleza",

    "Sir" : "Nobleza",

    "Mlle" : "Miss",

    "Col" : "Oficial",

    "Capt" : "Oficial",

    "Countess" : "Nobleza",

    "Jonkheer" : "Nobleza",

    "Dona" : "Nobleza"

}



tabla_completa["Titulo"] = tabla_completa["Name"].str.partition('.')[0].str.rpartition()[2].map(dic_titulos_simples)
grupo = tabla_completa.groupby(['Sex','Pclass','Titulo'])

tabla_completa['Age'] = grupo['Age'].apply(lambda x: x.fillna(x.median()))
tabla_completa['Fare'] = tabla_completa['Fare'].fillna(tabla_completa['Fare'].median())
tabla_completa['Num_Familiares'] = tabla_completa['SibSp'] + tabla_completa['Parch']
bins = [0,15,47,100]

labels = ['Menor','Intermedio','Mayor']

tabla_completa['Grupo_Edad'] = pd.cut(tabla_completa['Age'], bins=bins, labels=labels, right=True)
tabla_completa['Sex'] = tabla_completa['Sex'].map({"male": 0, "female":1})



# get dummies permite tener una columna individual para cada valor.

# Esto es preferible si no hay una forma obvia de asignarle a uno más valor que a otro

dummies_pclass = pd.get_dummies(tabla_completa['Pclass'], prefix="Pclass")

dummies_titulo = pd.get_dummies(tabla_completa['Titulo'], prefix="Titulo")

dummies_letra_cab = pd.get_dummies(tabla_completa['Letra_Cabina'], prefix="Letra_Cabina")

dummies_grupo_edad = pd.get_dummies(tabla_completa['Grupo_Edad'], prefix="Grupo_Edad")



dummies_tabla = pd.concat([tabla_completa, dummies_pclass, dummies_titulo, dummies_letra_cab,dummies_grupo_edad], axis=1)



# Eliminar atributos categoricos

dummies_tabla.drop(['Pclass', 'Titulo', 'Cabin', 'Letra_Cabina', 'Grupo_Edad', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
# Visualizar nueva tabla

dummies_tabla.head()
train_df = dummies_tabla[ :len(train_df)]

test_df = dummies_tabla[(len(dummies_tabla) - len(test_df)): ]



# Regresar 'Survived' a enteros

train_df['Survived'] = train_df['Survived'].astype(int)



# Mostrar tablas

print("Train: \n",train_df)

print("Test: \n",test_df)
# Arreglar indices

#test_df.index.name = 'PassengerID'

test_df = test_df.reset_index(drop = True)

#train_df.index.name = 'PassengerId'

train_df = train_df.reset_index(drop = True)



# Separar train en X y Y

train_dfX = train_df.drop(['PassengerId','Survived'], axis=1)

train_dfY = train_df['Survived']

submission = pd.DataFrame(data=test_df['PassengerId'].copy())

#submission['PassengerId'] = test_df['PassengerId'].copy()

print(submission)

test_df = test_df.drop(['PassengerId', 'Survived'], axis=1)
sc = StandardScaler()

train_dfX = sc.fit_transform(train_dfX)

test_df = sc.transform(test_df)
# Se tienen 891 samples para training (relativamente pequeño)

# Dividirlo en 85% - 15% permite una apreciación en el error del 0,75% aproximadamente

train_dfX,val_dfX,train_dfY, val_dfY = train_test_split(train_dfX,train_dfY , test_size=0.1, stratify=train_dfY)

print("Entrnamiento: ",train_dfX.shape)

print("Validacion : ",val_dfX.shape)
precisiones_globales=[]

epochs = 40

def graf_model(train_history):

    f = plt.figure(figsize=(epochs,10))

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
def func_model(arquitectura): 

    np.random.seed(42) # Random seed fija para facilitar comparar modelos

    random_seed = 42

    first =True

    inp = Input(shape=(train_dfX.shape[1],))

    for capa in arquitectura:        

        if first:

            x=Dense(capa, activation="relu", kernel_initializer=initializers.RandomNormal(seed=random_seed), bias_initializer='zeros')(inp)            

            first = False

        else:

            x=Dense(capa, activation="relu", kernel_initializer=initializers.RandomNormal(seed=random_seed), bias_initializer='zeros')(x)  

    x=Dense(1, activation="sigmoid", kernel_initializer=initializers.RandomNormal(seed=random_seed), bias_initializer='zeros')(x)  

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.0002), metrics=['binary_accuracy'])

    return model
arq1 = [1024, 1024, 512]

model1 = None

model1 = func_model(arq1)

train_history_tam1 = model1.fit(train_dfX, train_dfY, batch_size=32, epochs=epochs, validation_data=(val_dfX, val_dfY))

graf_model(train_history_tam1)

precision(model1)
arq2 = [1024, 512, 512]

model2 = None

model2 = func_model(arq2)

train_history_tam2 = model2.fit(train_dfX, train_dfY, batch_size=32, epochs=epochs, validation_data=(val_dfX, val_dfY))

graf_model(train_history_tam2)

precision(model2)
arqFinal = [1024, 1024, 1024]

modelF = None

modelF = func_model(arqFinal)

print(modelF.summary())

train_history_tamF = modelF.fit(train_dfX, train_dfY, batch_size=32, epochs=epochs, validation_data=(val_dfX, val_dfY))

graf_model(train_history_tamF)

precision(modelF, True)
def func_model_reg(): 

    np.random.seed(42) # Random seed fija para facilitar comparar modelos

    random_seed = 42

    inp = Input(shape=(train_dfX.shape[1],))

    x=Dropout(0.1)(inp)

    x=Dense(1024, activation="relu", kernel_initializer=initializers.RandomNormal(seed=random_seed), bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01))(x)

    x=Dropout(0.7)(x)

    x=Dense(1024, activation="relu", kernel_initializer=initializers.RandomNormal(seed=random_seed), bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01))(x)

    x=Dropout(0.7)(x)

    x=Dense(512, activation="relu", kernel_initializer=initializers.RandomNormal(seed=random_seed), bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01))(x)

    x=Dropout(0.9)(x)  

    x=Dense(1, activation="sigmoid", kernel_initializer=initializers.RandomNormal(seed=random_seed), bias_initializer='zeros')(x) 

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.0002), metrics=['binary_accuracy'])

    return model
modelReg = None

modelReg = func_model_reg()

train_history_tamReg = modelReg.fit(train_dfX, train_dfY, batch_size=32, epochs=epochs, validation_data=(val_dfX, val_dfY))

graf_model(train_history_tamReg)

precision(modelReg)
y_test = modelReg.predict(test_df)

submission['Survived'] = y_test.round().astype(int)

submission.to_csv('submission.csv', index=False)