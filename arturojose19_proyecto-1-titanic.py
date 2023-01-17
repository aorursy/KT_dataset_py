import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





# Red Neuronal

import keras 

from keras.models import Sequential # intitialize the ANN

from keras.layers import Dense      # create layers

from keras.callbacks import ReduceLROnPlateau

from keras.layers import Input, Dense, BatchNormalization, Add, GaussianNoise, Dropout

from keras.models import Model



# Regresor Forestal Random

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



# Cargar Datos

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df = df_train.append(df_test , ignore_index = True)



# Inspreccion 

df_train.shape, df_test.shape, df_train.columns.values



# Ignora advertencia para solucionar error en graficos

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print("Train shape : ",train_df.shape)

print("Test shape : ",test_df.shape)
# verifica si hay algu NAN

df['Pclass'].isnull().sum(axis=0)
# Verifica la correlación entre Pclass y Survived

df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
df.Name
df['Title'] = df.Name.map( lambda x: x.split(',')[1].split( '.' )[0].strip())



# Cantidad de personas por Titulo

df['Title'].value_counts()
df['Title'] = df['Title'].replace('Mlle', 'Miss')

df['Title'] = df['Title'].replace(['Mme','Lady','Ms'], 'Mrs')

df.Title.loc[ (df.Title !=  'Master') & (df.Title !=  'Mr') & (df.Title !=  'Miss') 

             & (df.Title !=  'Mrs')] = 'Others'



# Verifica la correlación entre Título y los que se salvaron

df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# cantidad de personas para cada título

df['Title'].value_counts()
df = pd.concat([df, pd.get_dummies(df['Title'])], axis=1).drop(labels=['Name'], axis=1)
# verifica si hay algu NAN

df.Sex.isnull().sum(axis=0)
# correlación entre sexo y sobrevivido

df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
# mapea los dos géneros a 0 y 1

df.Sex = df.Sex.map({'male':0, 'female':1})
# verifica si hay algu NAN

df.Age.isnull().sum(axis=0)
# verifica si hay algu NAN

df.SibSp.isnull().sum(axis=0), df.Parch.isnull().sum(axis=0)
#crea una nueva característica "Familia"

df['Family'] = df['SibSp'] + df['Parch'] + 1



#correlación entre familia y sobrevivido

df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()
#cantidad de personas para cada tamaño de familia

df['Family'].value_counts()
df.Family = df.Family.map(lambda x: 0 if x > 4 else x)

#Una función lambda puede tomar cualquier número de argumentos, pero solo puede tener una expresión.

df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()
df['Family'].value_counts()
# verifica si hay algu NAN

df.Ticket.isnull().sum(axis=0)
df.Ticket.head(20)
df.Ticket = df.Ticket.map(lambda x: x[0])



# correlación entre Ticket y Survived

df[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()
# cantidad de personas para cada tipo de entradas

df['Ticket'].value_counts()
df[['Ticket', 'Fare']].groupby(['Ticket'], as_index=False).mean()
df[['Ticket', 'Pclass']].groupby(['Ticket'], as_index=False).mean()
# verifica si hay algu NAN

df.Fare.isnull().sum(axis=0)
df.Ticket[df.Fare.isnull()]
df.Pclass[df.Fare.isnull()]
df.Cabin[df.Fare.isnull()]
df.Embarked[df.Fare.isnull()]
# boxplot para visualizar la distribución de la tarifa para cada Pclass

sns.boxplot('Pclass','Fare',data=df)

plt.ylim(0, 300) # ignorar un punto de datos con tarifa> 500

plt.show()
# correlación entre Pclass y Fare

df[['Pclass', 'Fare']].groupby(['Pclass']).mean()
# Se divide la desviación estándar por la media. Una relación más baja seria mas precisa

# distribución de la tarifa en cada clase

df[['Pclass', 'Fare']].groupby(['Pclass']).std() / df[['Pclass', 'Fare']].groupby(['Pclass']).mean()
# boxplot para visualizar la distribución de la tarifa para cada boleto

sns.boxplot('Ticket','Fare',data=df)

plt.ylim(0, 300) # ignorar un punto de datos con tarifa> 500

plt.show()
# Repetimos el proceso

df[['Ticket', 'Fare']].groupby(['Ticket']).mean()
# Repetimos el proceso

df[['Ticket', 'Fare']].groupby(['Ticket']).std() /  df[['Ticket', 'Fare']].groupby(['Ticket']).mean()
# usamos boxplot

sns.boxplot('Embarked','Fare',data=df)

plt.ylim(0, 300) 

plt.show()
# correlación entre Embarque y Tarifa

df[['Embarked', 'Fare']].groupby(['Embarked']).mean()
# mismo procedimiento que antes

df[['Embarked', 'Fare']].groupby(['Embarked']).std() /  df[['Embarked', 'Fare']].groupby(['Embarked']).mean()
guess_Fare = df.Fare.loc[ (df.Ticket == '3') & (df.Pclass == 3) & (df.Embarked == 'S')].median()

df.Fare.fillna(guess_Fare , inplace=True)



# los valores medios de tarifa para las personas que murieron y sobrevivieron

df[['Fare', 'Survived']].groupby(['Survived'],as_index=False).mean()
# visualizar la distribución de la tarifa para las personas que sobrevivieron y murieron

grid = sns.FacetGrid(df, hue='Survived', size=4, aspect=1.5)

grid.map(plt.hist, 'Fare', alpha=.5, bins=range(0,210,10))

grid.add_legend()

plt.show()
# visualizar la correlación entre Fare y Survived usando un gráfico de dispersión

df[['Fare', 'Survived']].groupby(['Fare'],as_index=False).mean().plot.scatter('Fare','Survived')

plt.show()
# intervalos con igual cantidad de personas

df['Fare-bin'] = pd.qcut(df.Fare,5,labels=[1,2,3,4,5]).astype(int)



# correlación entre Fare-bin y Survived

df[['Fare-bin', 'Survived']].groupby(['Fare-bin'], as_index=False).mean()
# verifica si hay algu NAN

df.Cabin.isnull().sum(axis=0)
df = df.drop(labels=['Cabin'], axis=1)
# verifica si hay algu NAN

df.Embarked.isnull().sum(axis=0)
df.describe(include=['O']) # S es el más común
# llenar los NAN

df.Embarked.fillna('S' , inplace=True )
# correlación entre Embarcado y Sobrevivido

df[['Embarked', 'Survived','Pclass','Fare', 'Age', 'Sex']].groupby(['Embarked'], as_index=False).mean()
df = df.drop(labels='Embarked', axis=1)
# Visualiza la correlación entre Título y Edad.

grid = sns.FacetGrid(df, col='Title', size=3, aspect=0.8, sharey=False)

grid.map(plt.hist, 'Age', alpha=.5, bins=range(0,105,5))

plt.show()
# la edad promedio para cada título

df[['Title', 'Age']].groupby(['Title']).mean()
df[['Title', 'Age']].groupby(['Title']).std()
#Visualiza la correlación entre Fare-bin y Age.

grid = sns.FacetGrid(df, col='Fare-bin', size=3, aspect=0.8, sharey=False)

grid.map(plt.hist, 'Age', alpha=.5, bins=range(0,105,5))

plt.show()
# inspeccionar la edad media para cada Fare-bin

df[['Fare-bin', 'Age']].groupby(['Fare-bin']).mean()
df[['Fare-bin', 'Age']].groupby(['Fare-bin']).std()
# Visualiza la correlación entre SibSp y Age.

grid = sns.FacetGrid(df, col='SibSp', col_wrap=4, size=3.0, aspect=0.8, sharey=False)

grid.map(plt.hist, 'Age', alpha=.5, bins=range(0,105,5))

plt.show()
# inspeccionar la edad media para cada SibSp

df[['SibSp', 'Age']].groupby(['SibSp']).mean()
# inspeccionar la desviación estándar de la edad para cada SibSp

df[['SibSp', 'Age']].groupby(['SibSp']).std()
# Visualiza la correlación entre Parch y Age.

grid = sns.FacetGrid(df, col='Parch', col_wrap=4, size=3.0, aspect=0.8, sharey=False)

grid.map(plt.hist, 'Age', alpha=.5, bins=range(0,105,5))

plt.show()
df[['Parch', 'Age']].groupby(['Parch']).mean()
df[['Parch', 'Age']].groupby(['Parch']).std() 
df_sub = df[['Age','Master','Miss','Mr','Mrs','Others','Fare-bin','SibSp']]



X_train  = df_sub.dropna().drop('Age', axis=1)

y_train  = df['Age'].dropna()

X_test = df_sub.loc[np.isnan(df.Age)].drop('Age', axis=1)



regressor = RandomForestRegressor(n_estimators = 300)

regressor.fit(X_train, y_train)

y_pred = np.round(regressor.predict(X_test),1)

df.Age.loc[df.Age.isnull()] = y_pred



df.Age.isnull().sum(axis=0) # no hay mas NAN
bins = [ 0, 4, 12, 18, 30, 50, 65, 100]

age_index = (1,2,3,4,5,6,7)



df['Age-bin'] = pd.cut(df.Age, bins, labels=age_index).astype(int)



df[['Age-bin', 'Survived']].groupby(['Age-bin'],as_index=False).mean()
df[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()
df['Ticket'].value_counts()
df['Ticket'] = df['Ticket'].replace(['A','W','F','L','5','6','7','8','9'], '4')



# verifica la correlación

df[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()
df = pd.get_dummies(df,columns=['Ticket'])
# parámetros del gráfico

fig = plt.figure(figsize=(18,6), dpi=1600) 

alpha=alpha_scatterplot = 0.2 

alpha_bar_chart = 0.55



ax1 = plt.subplot2grid((2,3),(0,0))

# traza un gráfico de barras de los que sobrevivieron frente a los que no sobrevivieron.               

df.Survived.value_counts().plot(kind='bar', alpha=alpha_bar_chart)

# Esto establece muy bien los márgenes en matplotlib 

ax1.set_xlim(-1, 2)

# Titulo

plt.title("Distribución de la supervivencia. (1 = Sobrevivió)")    
# parámetros del gráfico

fig = plt.figure(figsize=(18,6), dpi=1600) 

alpha=alpha_scatterplot = 0.2 

alpha_bar_chart = 0.55



ax2 = plt.subplot2grid((2,3),(0,2))

df.Pclass.value_counts().plot(kind="barh", alpha=alpha_bar_chart)

ax2.set_ylim(-1, len(df.Pclass.value_counts()))

plt.title("Distribución de clase")
# parámetros del gráfico

fig = plt.figure(figsize=(18,6), dpi=1600) 

alpha=alpha_scatterplot = 0.2 

alpha_bar_chart = 0.55



plt.subplot2grid((2,3),(1,0), colspan=2)

df.Age[df.Pclass == 1].plot(kind='kde')    

df.Age[df.Pclass == 2].plot(kind='kde')

df.Age[df.Pclass == 3].plot(kind='kde')

plt.xlabel("Age")    

plt.title("Distribución por edad dentro de las clases")

plt.legend(('1era Clase', '2da Clase','3ra Clase'),loc='best') 
df = df.drop(labels=['SibSp','Parch','Age','Fare','Title'], axis=1)

y_train = df[0:891]['Survived'].values

X_train = df[0:891].drop(['Survived','PassengerId'], axis=1).values

X_test  = df[891:].drop(['Survived','PassengerId'], axis=1).values
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

epochs = 200

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
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
print("Train: ",X_train.shape)

print("Test : ",X_test.shape)
X_train,val_dfX,y_train, val_dfY = train_test_split(X_train,y_train , test_size=0.2, stratify=y_train)

print("Train: ",X_train.shape)

print("Test : ",val_dfX.shape)
def func_model():   

    inp = Input(shape=(17,))

    x=Dense(9, activation="relu", kernel_initializer='uniform', bias_initializer='zeros')(inp)

    x=Dense(9, activation="relu", kernel_initializer='uniform', bias_initializer='zeros')(x)

    x=Dense(5, activation="relu", kernel_initializer='uniform', bias_initializer='zeros')(x)

    x=Dense(1, activation="sigmoid", kernel_initializer='uniform', bias_initializer='zeros')(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

    return model

model = func_model()

print(model.summary())



train_history = model.fit(X_train, y_train, batch_size=512, epochs=epochs, validation_data=(val_dfX, val_dfY))
graf_model(train_history)

precision(model, True)
y_pred = model.predict(X_test)

y_final = (y_pred > 0.5).astype(int).reshape(X_test.shape[0])



output = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_final})

output.to_csv('prediction-ann.csv', index=False)