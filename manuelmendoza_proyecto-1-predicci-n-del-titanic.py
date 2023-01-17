import numpy as np               # linear algebra

import pandas as pd              # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt  # graphics

import seaborn as sns            # statistics visualization

sns.set()

%matplotlib inline
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



datasets = [train, test]

df = pd.concat(datasets, sort=False)
print('Train set info:')

train.info()



print('-' *50)



print('Test set info:')

test.info()
print('Train set nulls:', train.isnull().sum(), sep='\n')



print('-' *30)



print('Test set nulls:', test.isnull().sum(), sep='\n')
submission = test[['PassengerId']].copy()
df['Title'] = df.Name.map( lambda x: x.split(',')[1].split( '.' )[0].strip())



df['Title'].value_counts()
df['Title'] = df['Title'].replace(['Dr','Rev','the Countess','Jonkheer','Lady','Sir', 'Don','Dona'],'Nobles')

df['Title'] = df['Title'].replace(['Ms','Mlle'],'Miss')

df['Title'] = df['Title'].replace(['Mme'],'Mrs')



df['Title'].value_counts()
df['Title'] = df['Title'].replace(['Col','Major','Capt'],'Navy')



df['Title'].value_counts()
sns.barplot(x = 'Title', y = 'Survived', data = df)
train['Title'] = train.Name.map( lambda x: x.split(',')[1].split( '.' )[0].strip())

train['Title'] = train['Title'].replace(['Dr','Rev','the Countess','Jonkheer','Lady','Sir', 'Don','Dona'],'Nobles')

train['Title'] = train['Title'].replace(['Ms','Mlle'],'Miss')

train['Title'] = train['Title'].replace(['Mme'],'Mrs')

train['Title'] = train['Title'].replace(['Col','Major','Capt'],'Navy')



test['Title'] = test.Name.map( lambda x: x.split(',')[1].split( '.' )[0].strip())

test['Title'] = test['Title'].replace(['Dr','Rev','the Countess','Jonkheer','Lady','Sir', 'Don','Dona'],'Nobles')

test['Title'] = test['Title'].replace(['Ms','Mlle'],'Miss')

test['Title'] = test['Title'].replace(['Mme'],'Mrs')

test['Title'] = test['Title'].replace(['Col','Major','Capt'],'Navy')



print("Train set:", train['Title'].value_counts(), sep="\n")

print('-'*40)

print("Test set:", test['Title'].value_counts(), sep="\n")
test[test['Fare'].isnull()]
new_fare = df.groupby(['Pclass','Embarked']).Fare.median()[3]['S']

test['Fare'] = test['Fare'].fillna(new_fare)



test[test['PassengerId'] == 1044]
test.Fare.isnull().sum(axis=0)
train[train['Embarked'].isnull()]
figure, myaxis = plt.subplots(figsize=(10, 7.5))



sns.countplot(x = "Pclass", 

              hue="Embarked",

              data = train, 

              linewidth=2, 

              palette = {'S':"green", 'C':"blue", 'Q':'red'}, 

              ax = myaxis,

              color = 'white')



myaxis.set_title("Cantidad de Personas por Ciudad y Clase ", fontsize = 20, color = 'white')

myaxis.set_xlabel("Clase", fontsize = 15, color = 'white');

myaxis.set_ylabel("Cantidad por ciudad", fontsize = 15, color = 'white')

myaxis.legend(["S", "C", "Q"], loc = 'upper right')
total = df[df['Pclass'] == 1]



amount = total['PassengerId'].count()

st = total[total['Embarked'] == 'S']['PassengerId'].count()



print("Southampton:")

print(st / amount)
df[df['Cabin'] == 'B28']
train['Embarked'] = train['Embarked'].fillna('S')



train[train['PassengerId'] == 62]
train[train['PassengerId'] == 830]
grid = sns.FacetGrid(df, col='Title', size=3, aspect=0.8, sharey=False)

grid.map(plt.hist, 'Age', alpha=.5, bins=range(0,105,5))

plt.show()
for x in range(1,4):

    print("Clase " + str(x))

    print(df[df['Pclass'] == x]['Title'].value_counts())
for x in range(1,4):

    print("Clase " + str(x))

    grid = sns.FacetGrid(df[df['Pclass'] == x], col='Title', size=3, aspect=0.8, sharey=False)

    grid.map(plt.hist, 'Age', alpha=.5, bins=range(0,105,5))

    plt.show()
train["Age"].fillna(train.groupby(["Pclass","Title"])["Age"].transform("median"), inplace=True)

test["Age"].fillna(test.groupby(["Pclass","Title"])["Age"].transform("median"), inplace=True)
train["Cabin"] = train["Cabin"].fillna("M")

test["Cabin"] = test["Cabin"].fillna("M")



train.Cabin.isnull().sum()
train['Cabin'] = train.Cabin.map(lambda x: str(x)[0])

test['Cabin'] = test.Cabin.map(lambda x: str(x)[0])

print("Train Cabinas:", train['Cabin'].value_counts(), sep="\n")

print("Test Cabinas", test['Cabin'].value_counts(), sep="\n")
train["Cabin"] = train["Cabin"].replace(["T"], "A")

train["Cabin"] = train["Cabin"].replace(["G"], "F")

test["Cabin"] = test["Cabin"].replace(["T"], "A")

test["Cabin"] = test["Cabin"].replace(["G"], "F")



print("Train Cabinas:", train['Cabin'].value_counts(), sep="\n")

print("Test Cabinas", test['Cabin'].value_counts(), sep="\n")
print('Train set nulls:', train.isnull().sum(), sep='\n')



print('-' *30)



print('Test set nulls:', test.isnull().sum(), sep='\n')
print('Numero de valores únicos para Nombre:', len(train.Name.unique()))

print('Numero de valores únicos para los IDs:', len(train.PassengerId.unique()))

print('Numero de valores únicos para el precio:', len(train.Fare.unique()))

print('Numero de valores únicos para Clases:', len(train.Pclass.unique()))

print('Numero de valores únicos para numero de padres abordo:', len(train.Parch.unique()))

print('Numero de valores únicos para numero de hermanos abordo:', len(train.SibSp.unique()))

print('Numero de valores únicos para Embarcación:', len(train.Embarked.unique()))

print('Numero de valores únicos para Cabina:', len(train.Cabin.unique()))

print('Numero de valores únicos para Sexo:', len(train.Sex.unique()))

print('Numero de valores únicos para Ticket:', len(train.Ticket.unique()))
train = train.drop(labels=["Name","Ticket","PassengerId"], axis = 1)

test = test.drop(labels=["Name","Ticket","PassengerId"], axis = 1)
print("Persona más joven:", df.Age.min())

print("Persona más vieja ", df.Age.max())
bins = [ 0, 10, 20, 30, 40, 50, 60, 70, 80 ]

age_index = (1,2,3,4,5,6,7,8)

train['Age-Classes'] = pd.cut(train.Age, bins, labels = age_index).astype(int)

test['Age-Classes'] = pd.cut(test.Age, bins, labels = age_index).astype(int)



print("Train set:", train["Age-Classes"].value_counts(), sep="\n")

print("-"*20)

print("Test set", test["Age-Classes"].value_counts(), sep="\n")
train["Age-Classes"] = train["Age-Classes"].replace([8], 7)

test["Age-Classes"] = test["Age-Classes"].replace([8], 7)



train[['Age-Classes', 'Survived']].groupby(['Age-Classes'],as_index=False).mean()
print("Precio más bajo:", df.Fare.min())

print("Precio más alto:", df.Fare.max())
fare_index = [1,2,3,4,5]



train["Fare-Classes"] = pd.qcut(train.Fare, 5, labels = fare_index).astype(int)

test["Fare-Classes"] = pd.qcut(test.Fare, 5, labels = fare_index).astype(int)



print("Train set:", train["Fare-Classes"].value_counts(), sep="\n")

print("-"*20)

print("Test set", test["Fare-Classes"].value_counts(), sep="\n")
train[['Fare-Classes', 'Survived']].groupby(['Fare-Classes'], as_index=False).mean()
train["Family"] = train["SibSp"] + train["Parch"] + 1

test["Family"] = test["SibSp"] + test["Parch"] + 1



train['Family'].value_counts()
train["Family"] = train["Family"].replace([8,11], 7)

test["Family"] = test["Family"].replace([8,11], 7)



train['Family'].value_counts()
train = train.drop(labels=["SibSp","Parch"], axis = 1)

test = test.drop(labels=["SibSp","Parch"], axis = 1)



train[['Family', 'Survived']].groupby(['Family'],as_index=False).mean()
train = pd.get_dummies(train, columns=["Age-Classes"])

train = pd.get_dummies(train, columns=["Pclass"])

train = pd.get_dummies(train, columns=["Cabin"])

train = pd.get_dummies(train, columns=["Family"])

train = pd.get_dummies(train, columns=["Embarked"])

train = pd.get_dummies(train, columns=["Title"])

train = pd.get_dummies(train, columns=["Sex"])

train = pd.get_dummies(train, columns=["Fare-Classes"])



test = pd.get_dummies(test, columns=["Age-Classes"])

test = pd.get_dummies(test, columns=["Pclass"])

test = pd.get_dummies(test, columns=["Cabin"])

test = pd.get_dummies(test, columns=["Family"])

test = pd.get_dummies(test, columns=["Embarked"])

test = pd.get_dummies(test, columns=["Title"])

test = pd.get_dummies(test, columns=["Sex"])

test = pd.get_dummies(test, columns=["Fare-Classes"])



train.head()
train = train.drop(labels=["Age","Fare"], axis = 1)

test = test.drop(labels=["Age","Fare"], axis = 1)
# importando librerias

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
X_train = train.drop(labels=["Survived"], axis = 1)

Y_train = train["Survived"]



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

test = sc.fit_transform(test)
X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=0.20, stratify=Y_train)



print("Train set: ", X_train.shape)

print("Dev set: ", X_dev.shape)
from keras.layers import Input, Dense, BatchNormalization, Add, GaussianNoise, Dropout

from keras.models import Model

from sklearn.metrics import roc_auc_score

from keras.callbacks import ReduceLROnPlateau

from keras import regularizers

import matplotlib.pyplot as plt

from matplotlib.pyplot import *
def func_model(): 

    # input layer

    X = Input(shape=(40,))

    a = Dropout(0)(X)

    # input layer

    a = Dense(80, activation = "relu", kernel_initializer="glorot_normal",  kernel_regularizer = None)(a)

    a = Dropout(0)(a)

    # first hidden layer

    a = Dense(60, activation = "relu", kernel_initializer="glorot_normal",  kernel_regularizer = None)(a)

    a = Dropout(0)(a)

    # second hidden layer

    a = Dense(45, activation = "relu", kernel_initializer="glorot_normal",  kernel_regularizer = None)(a) 

    a = Dropout(0)(a)

    # third hidden layer

    a = Dense(30, activation = "relu", kernel_initializer="glorot_normal",  kernel_regularizer = None)(a) 

    a = Dropout(0)(a)

    # forth hidden layer

    a = Dense(15, activation = "relu", kernel_initializer="glorot_normal",  kernel_regularizer = None)(a) 

    a = Dropout(0)(a)

    # output layer

    y_hat=Dense(1, activation = "sigmoid")(a)

    

    model = Model(inputs = X, outputs = y_hat)

    model.compile(loss = 'binary_crossentropy', optimizer = 'Adam', metrics = ['binary_accuracy'])

    return model
def graf_model(train_history):

    f = plt.figure(figsize=(15,15))

    ax = f.add_subplot(121)

    ax2 = f.add_subplot(122)

    

    ax.plot(train_history.history['binary_accuracy'])

    ax.plot(train_history.history['val_binary_accuracy'])

    ax.set_title('model accuracy')

    ax.set_ylabel('accuracy')

    ax.set_xlabel('epoch')

    ax.legend(['train', 'test'], loc='upper left')

    

    ax2.plot(train_history.history['loss'])

    ax2.plot(train_history.history['val_loss'])

    ax2.set_title('model loss')

    ax2.set_ylabel('loss')

    ax2.set_xlabel('epoch')

    ax2.legend(['train', 'test'], loc='upper left')

    plt.show()
def precision(model, registrar=False):

    y_pred = model.predict(X_train)

    train_auc = roc_auc_score(Y_train, y_pred)

    y_pred = model.predict(X_dev)

    val_auc = roc_auc_score(Y_dev, y_pred)

    print('Train AUC: ', train_auc)

    print('Vali AUC: ', val_auc)

    if registrar:

        precisiones_globales.append([train_auc,val_auc])
modelNN = func_model()

epochs = 100



train_history = modelNN.fit(X_train, Y_train, batch_size=32, epochs=epochs, validation_data=(X_dev,Y_dev))
graf_model(train_history)
precision(modelNN)
def modelRegularized(): 

    # input layer

    X = Input(shape=(40,))

    a = Dropout(0)(X)

    # first hidden layer

    a = Dense(80, activation = "relu", kernel_initializer="glorot_normal",  kernel_regularizer = regularizers.l2(0.01))(a)

    a = Dropout(0.5)(a)

    # second hidden layer

    a = Dense(60, activation = "relu", kernel_initializer="glorot_normal",  kernel_regularizer = regularizers.l2(0.01))(a)

    a = Dropout(0.7)(a)

    # third hidden layer

    a = Dense(45, activation = "relu", kernel_initializer="glorot_normal",  kernel_regularizer = regularizers.l2(0.01))(a) 

    a = Dropout(0.5)(a)

    # forth hidden layer

    a = Dense(30, activation = "relu", kernel_initializer="glorot_normal",  kernel_regularizer = regularizers.l2(0.01))(a) 

    a = Dropout(0.2)(a)

    # fifth hidden layer

    a = Dense(15, activation = "relu", kernel_initializer="glorot_normal",  kernel_regularizer = regularizers.l2(0.01))(a) 

    a = Dropout(0)(a)

    # output layer

    y_hat=Dense(1, activation = "sigmoid")(a)

    

    model = Model(inputs = X, outputs = y_hat)

    model.compile(loss = 'binary_crossentropy', optimizer = 'Adam', metrics = ['binary_accuracy'])

    return model
start = time.time()

modelF = modelRegularized()



train_history = modelF.fit(X_train, Y_train, batch_size=64, epochs=130, validation_data=(X_dev, Y_dev), verbose=0)

graf_model(train_history)

precision(modelF)

timeRecord = time.time() - start

print("--- %s seconds ---" % (timeRecord))
y_test = modelF.predict(test)

for i in range(len(y_test)):

    if y_test[i] < 0.5: 

        y_test[i] = 0

    else:

        y_test[i] = 1

        

submission['Survived'] = y_test.astype(int)

submission.to_csv('submission.csv', index=False)