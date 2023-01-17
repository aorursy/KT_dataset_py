# linear algebra

import numpy as np 



# data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd   



# Input data 

import os

print(os.listdir("../input"))



#Medir tiempo de ejecución

import time



#Gráficos de barra

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()



#Modelo

from keras.layers import Input, Dense, BatchNormalization, Add, GaussianNoise, Dropout

from keras.models import Model

from sklearn.metrics import roc_auc_score

from tensorflow.keras.layers import Wrapper

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

from keras import regularizers, initializers, optimizers

# Feature Scaling

from sklearn.preprocessing import StandardScaler
train = pd.read_csv("../input/train.csv")     # leyendo data de entrenamiento 

test = pd.read_csv("../input/test.csv")       # leyendo data de prueba 

print("Train shape: ",train.shape)            # Tamaño del data set (filas,columnas)

print("Test shape: ",test.shape)              # La columna que falta en el set test, es lo que tenemos que predecir (Survived)
#Formato de los datos del train set, podemos ver Name, sex, Parch, SibSp, Ticket Embarked y Cabin no son númericos 

train.info()  
#Formato de los datos del test set, podemos ver Name, sex, Parch, SibSp, Ticket Embarked y Cabin no son númericos 

test.info()
train.head(80)
#Muestra la cantidad de datos que faltan por columna en el train set, especialmente en la columna Cabin 

train.isnull().sum() 
#Muestra la cantidad de datos que faltan por columna en el test set, especialmente en la columna Cabin y Age

test.isnull().sum()
#Función auxiliar para imprimir gráficos de barra

def bar_chart(feature):

    sobreviviente = train[train['Survived']==1][feature].value_counts()

    muerto = train[train['Survived']==0][feature].value_counts()

    df = pd.DataFrame([sobreviviente,muerto])

    df.index = ['Sobreviviente','Muerto']

    df.plot(kind='bar',stacked=True, figsize=(10,5))
bar_chart('Sex')
bar_chart('Embarked')
bar_chart('Parch')
bar_chart('Pclass')
bar_chart('SibSp')
train.head()
#Uniendo los datos de train y test pero pre-procesarlos más fácil y rápido.

datostotales = [train, test]     
#Creando columna Title extrayendo el título de la persona por su nombre.

for datatotal in datostotales:

    datatotal['Title'] = datatotal['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
#Ya que son demasiados títulos se agrupan del 0 al 3, para ayudar a generalizar.

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 

                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for datatotal in datostotales:

    datatotal['Title'] = datatotal['Title'].map(title_mapping)
train.head() #Se puede ver como se agrego la columna "Title" y se mapeo a valores númericos.
#Eliminando columna Name

train.drop('Name', axis=1, inplace=True)

test.drop('Name', axis=1, inplace=True)
#Mapeando el sexo a valores númericos siendo male: 0 y female: 1

sex_mapping = {"male": 0, "female": 1}

for datatotal in datostotales:

    datatotal['Sex'] = datatotal['Sex'].map(sex_mapping)
#Cambiando los valores nulos de la columna Age con la media de dicho campo agrupandolos por su edad y título

train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)

test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)
#train.head(30)

#Cambiando los valores de la columna Age con la media de dicho campo agrupandolos por su edad y título

train.groupby("Title")["Age"].transform("median")
#Creando intervalos para la la columna edad, para ayudar a generalizar mejor

for datatotal in datostotales:

    datatotal.loc[ datatotal['Age'] <= 15, 'Age'] = 0,                           # Funcionan como un condicional

    datatotal.loc[(datatotal['Age'] > 15) & (datatotal['Age'] <= 35), 'Age'] = 1,

    datatotal.loc[(datatotal['Age'] > 35) & (datatotal['Age'] <= 55), 'Age'] = 2,

    datatotal.loc[(datatotal['Age'] > 55) & (datatotal['Age'] <= 69), 'Age'] = 3,

    datatotal.loc[ datatotal['Age'] > 69, 'Age'] = 4
bar_chart('Age')
Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()

Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()

Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

df.index = ['1st class','2nd class', '3rd class']

df.plot(kind='bar',stacked=True, figsize=(10,5))
#Reemplazando por valores nulos con S, este de utiliza como valor por defecto

for datatotal in datostotales:

    datatotal['Embarked'] = datatotal['Embarked'].fillna('Q')
#Mapeando en que puerto embarcó el pasajero

embarked_mapping = {"S": 0, "C": 1, "Q": 2}

for datatotal in datostotales:

    datatotal['Embarked'] = datatotal['Embarked'].map(embarked_mapping)
#Cambiando los valores nulos de la columna Fare con la media de dicho campo agrupandolos por su Pclass y Fare

train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)

test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
#Estableciendo intervalos para la tarifa del pasaje  para ayudar a generalizar mejor

for datatotal in datostotales:

    datatotal.loc[ (datatotal['Fare'] <= 30), 'Fare'] = 0,

    datatotal.loc[ (datatotal['Fare'] > 30) & (datatotal['Fare'] <= 100), 'Fare'] = 1,

    datatotal.loc[ (datatotal['Fare'] > 30) & (datatotal['Fare'] <= 100), 'Fare'] = 2,

    datatotal.loc[ (datatotal['Fare'] > 100), 'Fare'] = 3
train.Cabin.value_counts()
#Extrayendo la primera letra de la columna Cabin 

for datatotal in datostotales:

    datatotal['Cabin'] = datatotal['Cabin'].str[:1]
#Mapeando los datos de la columna cabina

cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}

for datatotal in datostotales:

    datatotal['Cabin'] = datatotal['Cabin'].map(cabin_mapping)
#Remplazando los valores nulos de Cabin con la media de dicha columna, agrupondolos por Pclass y Cabin

train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
#Creando una columna FamilySize que mide el número total de parientes abordo que tenian los pasajeros

train["FamilySize"] = train["SibSp"] + train["Parch"] + 1

test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
#Mapeando de datos de tamaño de la familia

family_size = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}

for datatotal in datostotales:

    datatotal['FamilySize'] = datatotal['FamilySize'].map(family_size)
#Eliminando columnas Ticket, SibSp, Parch 

features_drop = ['Ticket', 'SibSp', 'Parch']

train = train.drop(features_drop, axis=1)

test = test.drop(features_drop, axis=1)

#Eliminando PassengerId 

train = train.drop(['PassengerId'], axis=1)
train.head()
train_dfX = train.drop('Survived', axis=1)

train_dfY = train['Survived']

submission = test[['PassengerId']].copy()

test_df = test.drop(['PassengerId'], axis=1)
#Haciendo One Hot Enconding 

categorical = ['Embarked', 'Title', 'Pclass', 'Fare']

for var in categorical:

    train_dfX = pd.concat([train_dfX, pd.get_dummies(train_dfX[var], prefix=var)], axis=1)

    del train_dfX[var]
#Haciendo One Hot Enconding 

categorical = ['Embarked', 'Title', 'Pclass', 'Fare'] 

for var in categorical:

    test_df = pd.concat([test_df, pd.get_dummies(test_df[var], prefix=var)], axis=1)

    del test_df[var]
#Tamaños vectores de entrenamiento y prueba

test_df.shape, train_dfX.shape
train_dfX.head()
test_df.head()
precisiones_globales=[]

epochs = 30 

def graf_model(train_history):

    f = plt.figure(figsize=(15,10))

    ax = f.add_subplot(121)

    ax2 = f.add_subplot(122)

    # summarize history for accuracy

    ax.plot(train_history.history['binary_accuracy'])

    ax.plot(train_history.history['val_binary_accuracy'])

    ax.set_title('Model Accuracy')

    ax.set_ylabel('Accuracy')

    ax.set_xlabel('Epoch')

    ax.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    ax2.plot(train_history.history['loss'])

    ax2.plot(train_history.history['val_loss'])

    ax2.set_title('Model Loss')

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

    print('Test AUC: ', val_auc)

    if registrar:

        precisiones_globales.append([train_auc,val_auc])
sc = StandardScaler()

train_dfX = sc.fit_transform(train_dfX)

test_df = sc.transform(test_df)

print("Test shape : ",test_df.shape)
train_dfX,val_dfX,train_dfY, val_dfY = train_test_split(train_dfX,train_dfY , test_size=0.10, stratify=train_dfY)

print("Tamaño set de Entrenamiento: ",train_dfX.shape)

print("Tamaño set de Validacion : ",val_dfX.shape)
def func_model():

    inp = Input(shape=(17,)) 

    x=Dropout(0.1)(inp)

    x=Dense(350, activation="relu", kernel_regularizer=regularizers.l2(0.01))(inp)

    x=Dropout(0.50)(x)

    x=Dense(350, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)

    x=Dropout(0.50)(x)

    x=Dense(350, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)

    x=Dropout(0.30)(x)

    x=Dense(350, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)

    x=Dropout(0.30)(x)

    x=Dense(350, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)

    x=Dropout(0)(x)

    x=Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['binary_accuracy'])

    return model
start_time = time.time()

model = func_model()

print(model.summary())
train_history = model.fit(train_dfX, train_dfY, batch_size=64, epochs=epochs, validation_data=(val_dfX, val_dfY))
graf_model(train_history)
precision(model, True)
#Tiempo de ejecución

print("Tiempo de ejecución %s segundos" % (time.time() - start_time))
y_test = model.predict(test_df)

submission['Survived'] = np.rint(y_test).astype(int)

print(submission)

submission.to_csv('submission.csv', index=False)
