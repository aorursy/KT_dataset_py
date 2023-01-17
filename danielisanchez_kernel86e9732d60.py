# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory







import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print("Train shape : ",train.shape)

print("Test shape : ",test.shape)
train.head()

test.head()
train.info()
test.info()
train.head(80)
train.isnull().sum()
test.isnull().sum()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()
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
datostotales = [train, test]

for datatotal in datostotales:

    datatotal['Title'] = datatotal['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'].value_counts()
test['Title'].value_counts()
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 

                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for datatotal in datostotales:

    datatotal['Title'] = datatotal['Title'].map(title_mapping)
train.head()
train.head(50)
test.head(50)
bar_chart('Title')
train.drop('Name', axis=1, inplace=True)

test.drop('Name', axis=1, inplace=True)
train.head(20)
test.head()
sex_mapping = {"male": 0, "female": 1}

for datatotal in datostotales:

    datatotal['Sex'] = datatotal['Sex'].map(sex_mapping)

    
bar_chart('Sex')
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)

test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)
train.head(30)

train.groupby("Title")["Age"].transform("median")
for datatotal in datostotales:

    datatotal.loc[ datatotal['Age'] <= 16, 'Age'] = 0,

    datatotal.loc[(datatotal['Age'] > 16) & (datatotal['Age'] <= 26), 'Age'] = 1,

    datatotal.loc[(datatotal['Age'] > 26) & (datatotal['Age'] <= 36), 'Age'] = 2,

    datatotal.loc[(datatotal['Age'] > 36) & (datatotal['Age'] <= 62), 'Age'] = 3,

    datatotal.loc[ datatotal['Age'] > 62, 'Age'] = 4
bar_chart('Age')

Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()

Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()

Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

df.index = ['1st class','2nd class', '3rd class']

df.plot(kind='bar',stacked=True, figsize=(10,5))
for datatotal in datostotales:

    datatotal['Embarked'] = datatotal['Embarked'].fillna('S')

embarked_mapping = {"S": 0, "C": 1, "Q": 2}

for datatotal in datostotales:

    datatotal['Embarked'] = datatotal['Embarked'].map(embarked_mapping)
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)

test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)

train.head(50)
for datatotal in datostotales:

    datatotal.loc[ datatotal['Fare'] <= 17, 'Fare'] = 0,

    datatotal.loc[(datatotal['Fare'] > 17) & (datatotal['Fare'] <= 30), 'Fare'] = 1,

    datatotal.loc[(datatotal['Fare'] > 30) & (datatotal['Fare'] <= 100), 'Fare'] = 2,

    datatotal.loc[ datatotal['Fare'] > 100, 'Fare'] = 3
train.Cabin.value_counts()
for datatotal in datostotales:

    datatotal['Cabin'] = datatotal['Cabin'].str[:1]
Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()

Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()

Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

df.index = ['1st class','2nd class', '3rd class']

df.plot(kind='bar',stacked=True, figsize=(10,5))
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}

for datatotal in datostotales:

    datatotal['Cabin'] = datatotal['Cabin'].map(cabin_mapping)
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1

test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}

for datatotal in datostotales:

    datatotal['FamilySize'] = datatotal['FamilySize'].map(family_mapping)
features_drop = ['Ticket', 'SibSp', 'Parch']

train = train.drop(features_drop, axis=1)

test = test.drop(features_drop, axis=1)

train = train.drop(['PassengerId'], axis=1)
train.head()
train_dfX = train.drop('Survived', axis=1)

train_dfY = train['Survived']

submission = test[['PassengerId']].copy()

test_df = test.drop(['PassengerId'], axis=1)
#Nuevo 

categorical = ['Embarked', 'Title']



for var in categorical:

    train_dfX = pd.concat([train_dfX, 

                    pd.get_dummies(train_dfX[var], prefix=var)], axis=1)

    del train_dfX[var]
#Nuevo

categorical = ['Embarked', 'Title']



for var in categorical:

    test_df = pd.concat([test_df, 

                    pd.get_dummies(test_df[var], prefix=var)], axis=1)

    del test_df[var]
test_df.shape, train_dfX.shape
test_df.head()
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
#Numero de epochs 25 ha sido el mejor, probemos con 19

precisiones_globales=[]

epochs = 25

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
sc = StandardScaler()

train_dfX = sc.fit_transform(train_dfX)

test_df = sc.transform(test_df)

print("Test shape : ",test_df.shape)
train_dfX,val_dfX,train_dfY, val_dfY = train_test_split(train_dfX,train_dfY , test_size=0.1, stratify=train_dfY)

print("Entrnamiento: ",train_dfX.shape)

print("Validacion : ",val_dfX.shape)
#Mejores resultados con dos capas una de 100 y otra de 90 y dropuot de 0.5

def func_model():   

    inp = Input(shape=(13,))

    x=Dropout(0.1)(inp)

    x=Dense(500, activation="relu", kernel_regularizer=regularizers.l2(0.01))(inp)

    x=Dropout(0.4)(x)

    x=Dense(500, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x) 

    x=Dropout(0.4)(x)

    x=Dense(500, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x) 

    x=Dropout(0.3)(x)

    x=Dense(500, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x) 

    x=Dropout(0.3)(x)

    x=Dense(500, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x) 

    x=Dropout(0.2)(x)

    x=Dense(500, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x) 

    x=Dropout(0)(x)

    x=Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['binary_accuracy'])

    return model

model = func_model()

print(model.summary())
train_history = model.fit(train_dfX, train_dfY, batch_size=64, epochs=epochs, validation_data=(val_dfX, val_dfY))
graf_model(train_history)
precision(model, True)
y_test = model.predict(test_df)

submission['Survived'] = np.rint(y_test).astype(int)

print(submission)

submission.to_csv('submission.csv', index=False)
