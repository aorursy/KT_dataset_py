# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





# Any results you write to the current directory are saved as output.
# Importar datos 

train= pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

p=pd.read_csv("../input/titanic/train.csv")

print("Train shape : ",train.shape)

print("Test shape : ",test.shape)
train.isnull().sum()
test.isnull().sum()
def bar_chart(feature):

    survived = train[train['Survived']==1][feature].value_counts()

    dead = train[train['Survived']==0][feature].value_counts()

    df2 = pd.DataFrame([survived,dead])

    df2.index = ['Survived','Dead']

    df2.plot(kind='bar',stacked=True, figsize=(10,5))
#separar de los nombres el titulo que posee cada persona y agregarlo como un nuevo dato a los sets

combinado = [train,test]

for dataset in combinado:

    dataset ['Titulo']= dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)



train['Titulo'].value_counts()
#Agrupar los Titulos en categorias mas generales 

for dataset in combinado:

    dataset['Titulo'] = dataset['Titulo'].replace([ 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Otros')

    dataset['Titulo'] = dataset['Titulo'].replace(['Countess', 'Lady', 'Sir'], 'Otros')

    dataset['Titulo'] = dataset['Titulo'].replace('Mlle', 'Miss')

    dataset['Titulo'] = dataset['Titulo'].replace('Ms', 'Miss')

    dataset['Titulo'] = dataset['Titulo'].replace('Mme', 'Mrs')

    dataset['Titulo'] = dataset['Titulo'].fillna('Otros')

    

train[['Titulo', 'Survived']].groupby(['Titulo'], as_index=False).mean()
bar_chart('Titulo')
#One Hot Encoding por cada Titulo  

train = pd.concat([train, pd.get_dummies(train['Titulo'])], axis=1)

test = pd.concat([test, pd.get_dummies(test['Titulo'])], axis=1)





train=train.drop(labels=['Name','Titulo'], axis=1) 

test=test.drop(labels=['Name','Titulo'], axis=1)



#agregar columna de tiene Cabina para one hot encoding 

test['Tiene_Cabina'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

train['Tiene_Cabina'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)



train[['Tiene_Cabina', 'Survived']].groupby(['Tiene_Cabina'], as_index=False).mean()
bar_chart('Tiene_Cabina')
train=train.drop(labels=['Cabin'], axis=1)

test=test.drop(labels=['Cabin'], axis=1) 
#One Hot Encoding  para cada sexo   

train = pd.concat([train, pd.get_dummies(train['Sex'])], axis=1) 

test = pd.concat([test, pd.get_dummies(test['Sex'])], axis=1)



bar_chart('Sex')
train=train.drop(labels=['Sex'], axis=1)

test=test.drop(labels=['Sex'], axis=1)
#LLenar valores de enbarked faltantes con s:

train = train.fillna({"Embarked": "S"})



bar_chart('Embarked')
#One Hot Encoding  para cada sitio de embarque 

train = pd.concat([train, pd.get_dummies(train['Embarked'])], axis=1) 

test = pd.concat([test, pd.get_dummies(test['Embarked'])], axis=1)



train=train.drop(labels=['Embarked'], axis=1)

test=test.drop(labels=['Embarked'], axis=1)
#Quitar valores null de el dato edad

combinado = [train,test]

for dataset in combinado:

    media=dataset['Age'].mean()

    std=dataset['Age'].std()

    cantidad=dataset['Age'].isnull().sum()

    #Seleccionar una edad aleatoria con los datos disponibles del dataset entero

    edad= np.random.randint(media - std, media + std, size=cantidad) 

    #De los datos null se asigna a uno aleatorio la edad seleccionada

    dataset['Age'][np.isnan(dataset['Age'])] = edad

    dataset['Age'] = dataset['Age'].astype(int)



train=train.drop(labels=['Ticket'], axis=1)

test=test.drop(labels=['Ticket'], axis=1)
combinado = [train,test]

for dataset in combinado:

    dataset['Familia']=dataset['Parch']+dataset['SibSp']+1

train=train.drop(labels=['Parch','SibSp'], axis=1)

test=test.drop(labels=['Parch','SibSp'], axis=1)
train = pd.concat([train, pd.get_dummies(train['Pclass'])], axis=1) 

test = pd.concat([test, pd.get_dummies(test['Pclass'])], axis=1)

train=train.drop(labels=['Pclass'], axis=1)

test=test.drop(labels=['Pclass'], axis=1)

train.head()
train=train.drop(labels=['PassengerId'], axis=1)

test.head()
y_train= train['Survived']

X_train= train.drop(labels='Survived', axis=1)

X_test= test.drop(labels='PassengerId',axis=1)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test= sc.fit_transform(X_test)
X_train,val_dfX,y_train, val_dfY = train_test_split(X_train,y_train , test_size=0.10, stratify=y_train)

print("Entrenamiento: ",X_train.shape)

print("Validacion : ",val_dfX.shape)

from keras.layers import Input, Dense, BatchNormalization, Add, GaussianNoise, Dropout

from keras.models import Model

from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential

from sklearn.metrics import roc_auc_score
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

    y_pred = model.predict(X_train)

    train_auc = roc_auc_score(y_train, y_pred)

    y_pred = model.predict(val_dfX)

    val_auc = roc_auc_score(val_dfY, y_pred)

    print('Train AUC: ', train_auc)

    print('Vali AUC: ', val_auc)

    if registrar:

        precisiones_globales.append([train_auc,val_auc])
arquitectura=[16,18,16,14,12]     

first =True

inp = Input(shape=(17,))

for capa in arquitectura:        

    if first:

        x=Dense(capa, activation="relu", kernel_initializer='glorot_uniform', bias_initializer='zeros')(inp)            

        first = False

    else:

        x=Dense(capa, activation="relu", kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)  

x=Dense(1, activation="sigmoid", kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)  

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['binary_accuracy'])



print(model.summary())



train_history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(val_dfX, val_dfY),verbose=2)
graf_model(train_history)
precision(model, True)
print("Entrenamiento: ",X_train.shape)

print("Validacion : ",val_dfX.shape)
def create_model_R():   

    inp = Input(shape=(17,))

    x=Dropout(0.1)(inp)

    x=Dense(20, activation="relu", kernel_initializer='glorot_normal', bias_initializer='zeros')(inp)

    x=Dropout(0.2)(x)

    x=Dense(18, activation="relu", kernel_initializer='glorot_normal', bias_initializer='zeros')(x)

    x=Dropout(0.2)(x)

    x=Dense(16, activation="relu", kernel_initializer='glorot_normal', bias_initializer='zeros')(x)

    x=Dropout(0.1)(x)

    x=Dense(14, activation="relu", kernel_initializer='glorot_normal', bias_initializer='zeros')(x)

    x=Dropout(0.1)(x)

    x=Dense(12, activation="relu", kernel_initializer='glorot_normal', bias_initializer='zeros')(x)

    x=Dropout(0.1)(x)

    x=Dense(1, activation="sigmoid", kernel_initializer='glorot_normal', bias_initializer='zeros')(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['binary_accuracy'])

    return model

model_R = create_model_R()

print(model_R.summary())
train_history_R = model_R.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(val_dfX, val_dfY),verbose=2)
graf_model(train_history_R)
precision(model_R, True)
y_pred = model_R.predict(X_test)



y_final = (y_pred > 0.5).astype(int).reshape(X_test.shape[0])



output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_final})

output.to_csv('prediction.csv', index=False)

pred = pd.read_csv('prediction.csv')

pred.head()
pred.head(418)