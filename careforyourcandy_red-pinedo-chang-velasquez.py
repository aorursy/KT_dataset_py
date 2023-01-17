# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #data visualization 

#Keras imports for machine learning

from keras.layers import Input, Dense, BatchNormalization, Add, GaussianNoise, Dropout

from keras.models import Model

from keras.layers import Wrapper

from keras.callbacks import ReduceLROnPlateau

from keras.utils import to_categorical

from keras import regularizers, Sequential

from keras.wrappers.scikit_learn import KerasClassifier

#sklearn imports for metrics and data division

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
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
train_set = pd.read_csv("../input/train.csv")

train_dfY = train_set['Survived']

test_set = pd.read_csv("../input/test.csv")

submission = test_set['PassengerId'].copy()

print(train_set.shape)

print(test_set.shape)

print(train_dfY.shape)





train_set['Cabin'].value_counts()

datasets = [train_set, test_set]

originalData = train_set

for dataset in datasets:

    dataset['Age'].fillna(dataset["Age"].median(), inplace=True)

    dataset['Embarked'].fillna(dataset["Embarked"].mode()[0], inplace=True)

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

    dataset.drop(['PassengerId','Cabin', 'Ticket'], axis=1, inplace=True)

    #+1 to count himself

    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1 

    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0

print('Ahora columnas null en el trainset')

print(train_set.isnull().sum())

print('Ahora columnas null en el test set')

print(test_set.isnull().sum())

for dataset in datasets:   

    #This creates an array with 0 beign the last name and 1 beign title + first and second name

    StringArray = dataset['Name'].str.split(", ", expand=True)

    #We grab the title and names and further split it into title and name.0 Contains the title, 1 contains the name

    StringArray = StringArray[1].str.split(".", expand=True)

    dataset['Title'] = StringArray[0]

    title_names = dataset['Title'].value_counts()

    print('===Prior to grouping===')

    print(title_names)

    #Podemos observar que hay una gran diferencia entre el número de Dr. y Master, agruparemos todo lo que sea menor o igual a 7 en titulos misc

    #También se podría hacer que cada título tenga su 

    title_names = (dataset['Title'].value_counts() > 7)

    dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == False else x)

    print('===Post grouping===')

    print(dataset['Title'].value_counts())

    
label = LabelEncoder()

enc = OneHotEncoder(handle_unknown='ignore')

train_set = datasets[0]

test_set = datasets[1]

train_set = train_set.drop(['Survived'], axis=1)

EncodedDataFrames = []

print(train_set.shape, test_set.shape)

wholeData = pd.concat([train_set, test_set], ignore_index=True)

#from 0 to 890: train_set. From 890 to 1308: Test_set



wholeData['Sex_Code'] = label.fit_transform(wholeData['Sex'])

wholeData['Embarked_Code'] = label.fit_transform(wholeData['Embarked'])

wholeData['Title_Code'] = label.fit_transform(wholeData['Title'])

#Column order: Female, Male

sex_encoded = to_categorical(wholeData['Sex_Code'])

SexOneHot = pd.DataFrame(sex_encoded, columns = ["Female", "Male"])

#Column order: C, Q ,S

embarked_encoded = to_categorical(wholeData['Embarked_Code'])

EmbarkedOneHot = pd.DataFrame(embarked_encoded, columns = ["C", "Q", "S"])

#Column order: Master, Misc, Miss, Mr, Mrs

title_encoded = to_categorical(wholeData['Title_Code']) 

TitleOneHot = pd.DataFrame(title_encoded, columns = ["Master", "Misc", "Miss", "Mr", "Mrs"])

#Column order: 0, 1, 2, 3

pclass_encoded = to_categorical(wholeData['Pclass'])

PClassOneHot = EmbarkedOneHot = pd.DataFrame(pclass_encoded, columns = ["0th Class", "1th Class", "2nd Class", "3rd Class"])

PClassOneHot = PClassOneHot.drop("0th Class", axis=1)

wholeData = pd.concat([wholeData, SexOneHot, EmbarkedOneHot, TitleOneHot, PClassOneHot], axis=1)

EncodedDataFrames.append(wholeData)



print(wholeData.shape)



#train_set = EncodedDataFrames[0]

#test_set = EncodedDataFrames[1]


wholeData = wholeData.drop(['Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'Pclass'], axis= 1)



[train_set, test_set] = np.split(wholeData, [891], axis= 0)

train_dfX = train_set

print(train_dfX.shape, train_dfY.shape)



#train_dfX = train_set.drop(['Sex_Code', 'Embarked_Code', 'Title_Code', 'Survived'], axis = 1)

train_dfX,val_dfX,train_dfY, val_dfY = train_test_split(train_dfX,train_dfY , test_size=0.1, stratify=train_dfY)



print("Entrenamiento: ",train_dfX.shape, train_dfY.shape)

print("Validacion : ",val_dfX.shape, val_dfY.shape)

print("Test: ", test_set.shape)

'''

def func_model():   

    inp = Input(shape=(20,))

    x=Dense(1028, activation="relu", kernel_initializer='glorot_normal', bias_initializer='zeros')(inp)

    x=Dense(1028, activation="relu", kernel_initializer='glorot_normal', bias_initializer='zeros')(x) 

    x=Dense(1, activation="sigmoid", kernel_initializer='glorot_normal', bias_initializer='zeros')(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

    return model

model = func_model()

print(model.summary())

'''



#Probando otro modelo



model = Sequential()

model.add(Dense(40, kernel_initializer = 'glorot_normal', bias_initializer='zeros', activation = 'relu', kernel_regularizer=regularizers.l2(0.01), input_dim = 19))

model.add(Dropout(0.2))



model.add(Dense(20, kernel_initializer = 'glorot_normal', bias_initializer='zeros', activation = 'relu', kernel_regularizer=regularizers.l2(0.01) ))

model.add(Dropout(0.2))



model.add(Dense(1, kernel_initializer = 'glorot_normal', bias_initializer='zeros', activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

model.summary()



#train_history = model.fit(train_dfX, train_dfY, batch_size= 25, epochs=20, validation_data=(val_dfX, val_dfY))



#Probando otro modelo 

train_history = model.fit(train_dfX, train_dfY, batch_size=24, epochs= 20, validation_data=(val_dfX, val_dfY))
graf_model(train_history)



y_test = model.predict(test_set)

y_formatted = np.where(y_test > 0.5, 1, 0)

y_dataFrame = pd.DataFrame(np.ravel(y_formatted), columns=['Survived'])

submission = pd.concat([submission, y_dataFrame], axis=1)

submission.to_csv('submission.csv', index=False)