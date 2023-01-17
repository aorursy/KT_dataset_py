import numpy as np #libreria calculos de algebra lineal

import pandas as pd  #libreria para hacer dataframes y preprocesar columnas y datos

import matplotlib.pyplot as plt #libreria para graficar

import seaborn as sns #visualizacion de estadisticas

sns.set() 

%matplotlib inline



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

#junto ambos dataframes en uno solo. Para aplicar las mismas modificaciones en ambos y luego cuando hacemos la red neuronal los volvemos a separar.

df = train.append(test , ignore_index = True)



#ver dimensiones y columnas

print(train.shape, test.shape, train.columns.values)
train.head(10) #muestro los diez primeros datos del dataset train
test.head(10)  #muestro los diez primeros datos del dataset test
df 
df.info()
df.isnull().sum()
train.isnull().sum()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set() # setting seaborn default for plots



def bar_chart(feature):

    survived = df[df['Survived']==1][feature].value_counts()

    dead = df[df['Survived']==0][feature].value_counts()

    df2 = pd.DataFrame([survived,dead])

    df2.index = ['Survived','Dead']

    df2.plot(kind='bar',stacked=True, figsize=(10,5))
# Se mapea a una variable booleana, si es hombre o mujer, no hizo falta usarla como categorica.

df.Sex = df.Sex.map({'male':0, 'female':1})
bar_chart('Sex') #Graficar sexo vs supervivencia.
bar_chart('Pclass')
# Agrupo por media la Supervivencia y la Clase de la persona

df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
df.Name.head(10)
#los títulos siempre se encuentran entre uba coma y un punto, buscamos los strings entre esos dos y lo guardamos en una nueva columna llamada Title.

df['Title'] = df.Name.map( lambda x: x.split(',')[1].split( '.' )[0].strip())



# inspect the amount of people for each title

df['Title'].value_counts()
df['Title'] = df['Title'].replace('Mlle', 'Miss')

df['Title'] = df['Title'].replace(['Mme','Lady','Ms'], 'Mrs')

df.Title.loc[ (df.Title !=  'Master') & (df.Title !=  'Mr') & (df.Title !=  'Miss') 

             & (df.Title !=  'Mrs')] = 'Others'



# correlacion del titulo con los sobrevivientes

df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
bar_chart('Title') #En la grafica se ve mejor que los Mr fueron los que mas murieron, pero los miss y mrs fueron los que mas sobrevivieron tambien
df['Title'].value_counts() #Contar numero de filas con los titulos
df = pd.concat([df, pd.get_dummies(df['Title'])], axis=1).drop(labels=['Name'], axis=1) #One hot enconding con las columnas
df.head() #Ver cambios en el dataframe
# Para crear una familia creo una nueva columna con los datos de los hermanos/esposos y padres/hijos mas la persona

df['Family'] = df['SibSp'] + df['Parch'] + 1
df.head() # ver la familia en el dataset
bar_chart('Family') 
# inspect the amount of people for each Family size

df['Family'].value_counts()
df.Family = df.Family.map(lambda x: 0 if x > 4 else x)

#Correlacion y media de familiariees que sobrevivieron por cantidad de familiares

df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()
df['Family'].value_counts()
df.Family.head(10) #ver dataset con la cantidad de miembros en las familias 
df.Ticket.head(20)
df.Ticket = df.Ticket.map(lambda x: x[0])



# Ver la correlacion entre ticket y supervivencia, tambien lo vemos en la grafica

df[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()
# Ver la cantidad de personas por cada tipo de entrada

df['Ticket'].value_counts()
# la media y correlacion del precio por cada tipo de tickets

df[['Ticket', 'Fare']].groupby(['Ticket'], as_index=False).mean()
#la media  y correlacion de la clase si es 1,2 o 3 por cada tipo de tickets

df[['Ticket', 'Pclass']].groupby(['Ticket'], as_index=False).mean()
# revisar si tiene valores nulos el precio

df.Fare.isnull().sum(axis=0)
# revisar cual ticket tiene ese precio nulo

df.Ticket[df.Fare.isnull()]
# revisar cual clase tienen ese precio nulo

df.Pclass[df.Fare.isnull()]
# revisar cual cabinas tienen ese precio nulo

df.Cabin[df.Fare.isnull()]
# revisar cual embarque tienen ese precio nulo

df.Embarked[df.Fare.isnull()]
#con loc puedo hacer la consula de los datos que necesito y le pongo la media

adivinarFare = df.Fare.loc[ (df.Ticket == '3') & (df.Pclass == 3) & (df.Embarked == 'S')].median()

df.Fare.fillna(adivinarFare , inplace=True)
# visualize los precios por las personas que murieron y sobrevivieron

grid = sns.FacetGrid(df, hue='Survived', size=4, aspect=1.5)

grid.map(plt.hist, 'Fare', alpha=.5, bins=range(0,210,10))

grid.add_legend()

plt.show()
# Dividir los precios en 5 intervalos

df['Fare-intervalo'] = pd.qcut(df.Fare,5,labels=[1,2,3,4,5]).astype(int)



# Calcular la media entre los sobrevientes por cada intervalo de precio

df[['Fare-intervalo', 'Survived']].groupby(['Fare-intervalo'], as_index=False).mean()
df.head()
#df = df.drop(labels=['Cabin'], axis=1)
df.Cabin.isnull().sum(axis=0)
df.Cabin.value_counts()
train_test_data = [df]

for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].str[:1]
Pclass1 = df[df['Pclass']==1]['Cabin'].value_counts()

Pclass2 = df[df['Pclass']==2]['Cabin'].value_counts()

Pclass3 = df[df['Pclass']==3]['Cabin'].value_counts()

df2 = pd.DataFrame([Pclass1, Pclass2, Pclass3])

df2.index = ['1st class','2nd class', '3rd class']

df2.plot(kind='bar',stacked=True, figsize=(10,5))
#Se procedio a mapear las categorias a una variable numerica y luego hacemos one hot encoding de ellas.

mapeo = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}

for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].map(mapeo)
#Para los valores null, se utilzo los promedios de cada clase para las cabinas faltantes

df["Cabin"].fillna(df.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

df.Cabin
df.Cabin.isnull().sum(axis=0)
# ver si hay valores nulos, hay 2 nulos

df.Embarked.isnull().sum(axis=0)
df.describe(include=['O']) # S is es el embarque mas comun
# Voy a llenar los Valores nulos con S, que es el mas comun

df.Embarked.fillna('S' , inplace=True )
# ver la media y correlacion entre embarque y supervivencia y la clase, precio, edad y sexo

df[['Embarked', 'Survived','Pclass','Fare', 'Age', 'Sex']].groupby(['Embarked'], as_index=False).mean()
Pclass1 = df[df['Pclass']==1]['Embarked'].value_counts()

Pclass2 = df[df['Pclass']==2]['Embarked'].value_counts()

Pclass3 = df[df['Pclass']==3]['Embarked'].value_counts()

df3 = pd.DataFrame([Pclass1, Pclass2, Pclass3])

df3.index = ['1st class','2nd class', '3rd class']

df3.plot(kind='bar',stacked=True, figsize=(10,5))
df.Age.isnull().sum(axis=0)
# Ver la correlacion de la edad con los titulos

grid = sns.FacetGrid(df, col='Title', size=3, aspect=0.8, sharey=False)

grid.map(plt.hist, 'Age', alpha=.5, bins=range(0,105,5))

plt.show()
# inspect the mean Age for each Title

df[['Title', 'Age']].groupby(['Title']).mean()
# los valores nulos los llenamos con la media de la edad de los titulos each title (Mr, Mrs, Miss, Master, Others)

df["Age"].fillna(df.groupby("Title")["Age"].transform("median"), inplace=True)
df.head(30)

df.groupby("Title")["Age"].transform("median")
df.Age.isnull().sum(axis=0) #ya no tengo variables nulas en la edad
bins = [ 0, 4, 12, 18, 30, 50, 65, 100] # Edades 

age_index = (1,2,3,4,5,6,7)  # ('bebe','nino','adolescente','joven','adulto','mayor','3raedad')

df['Age-intervalo'] = pd.cut(df.Age, bins, labels=age_index).astype(int)



#ver la correlacion si sobrevivio con los intervalos de edad

df[['Age-intervalo', 'Survived']].groupby(['Age-intervalo'],as_index=False).mean()
df[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()
df['Ticket'].value_counts()
df['Ticket'] = df['Ticket'].replace(['A','W','F','L','5','6','7','8','9'], '4')



# ver la media y correlacion con el cambio

df[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()
df = df.drop(labels=['Embarked'], axis=1)

df = df.drop(labels=['Cabin'], axis=1)


#One hot encoding

#df = pd.get_dummies(df,columns=['Family'])

df = pd.get_dummies(df,columns=['Fare-intervalo'])

df = pd.get_dummies(df,columns=['Age-intervalo'])

df = pd.get_dummies(df,columns=['Pclass'])

df = pd.get_dummies(df,columns=['Ticket'])

#df = pd.get_dummies(df,columns=['Cabin'])

#df = pd.get_dummies(df,columns=['Embarked'])



df.head() #dataset definitivo
df.isnull().sum(axis=0) #ver si quedaron todos los campos sin datos nulos
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



from matplotlib.pyplot import *
precisiones_globales=[]

epochs = 100

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
df = df.drop(labels=['SibSp','Parch','Age','Fare','Title'], axis=1) #borramos lo que no necesitamos, tengo los intervalos de edad e intervalos de fare

y_train = df[0:891]['Survived'].values

X_train = df[0:891].drop(['Survived','PassengerId'], axis=1).values

X_test  = df[891:].drop(['Survived','PassengerId'], axis=1).values
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
print("Entrnamiento: ",X_train.shape)

print("Test : ",X_test.shape)
X_train,val_dfX,y_train, val_dfY = train_test_split(X_train,y_train , test_size=0.20, stratify=y_train)

print("Entrenamiento: ",X_train.shape)

print("Validacion : ",val_dfX.shape)
def func_model():   

    inp = Input(shape=(29,))

    x=Dense(12, activation="relu", kernel_initializer='glorot_normal', bias_initializer='zeros')(inp)

    x=Dense(12, activation="relu", kernel_initializer='glorot_normal', bias_initializer='zeros')(x)

    x=Dense(14, activation="relu", kernel_initializer='glorot_normal', bias_initializer='zeros')(x)

    x=Dense(12, activation="relu", kernel_initializer='glorot_normal', bias_initializer='zeros')(x)

    x=Dense(1, activation="sigmoid", kernel_initializer='glorot_normal', bias_initializer='zeros')(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['binary_accuracy'])

    return model

model = func_model()

print(model.summary())


train_history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(val_dfX, val_dfY))
graf_model(train_history)

precision(model, True)
#np.random.seed(0) 



def func_model_reg():   

    inp = Input(shape=(29,))

    

    x=Dropout(0.1)(inp)

    

    x=Dense(12, activation="relu", kernel_initializer='glorot_normal', bias_initializer='zeros')(inp)

    x=Dropout(0.3)(x)

    

    x=Dense(12, activation="relu", kernel_initializer='glorot_normal', bias_initializer='zeros')(x)

    x=Dropout(0.3)(x)

    

    x=Dense(14, activation="relu", kernel_initializer='glorot_normal', bias_initializer='zeros')(x)

    x=Dropout(0.3)(x)

    

    x=Dense(12, activation="relu", kernel_initializer='glorot_normal', bias_initializer='zeros')(x)

    x=Dropout(0.1)(x)

    

    x=Dense(1, activation="sigmoid", kernel_initializer='glorot_normal', bias_initializer='zeros')(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['binary_accuracy'])

    return model

model = func_model_reg()

print(model.summary())





model1 = func_model_reg()

entrenadofinal = model1.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(val_dfX, val_dfY), verbose=0)
graf_model(entrenadofinal)

precision(model1)
## Ver la prediccion de los datos
y_pred = model1.predict(X_test)

y_final = (y_pred > 0.5).astype(int).reshape(X_test.shape[0])



output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_final})

output.to_csv('prediction.csv', index=False)
pred = pd.read_csv('prediction.csv')

pred.head()