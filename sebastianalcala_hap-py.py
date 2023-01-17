import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





# Red Neuronal

import keras 

from keras.models import Sequential 

from keras.layers import Dense    

from keras.callbacks import ReduceLROnPlateau

from keras.layers import Input, Dense, BatchNormalization, Add, GaussianNoise, Dropout

from keras.models import Model





import os

print(os.listdir("/kaggle/input/titanic")) #borrar despues





# Cargar Datos

df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

df = df_train.append(df_test , ignore_index = True, sort=False)



#Empezamos el preprocesamiento de la data

#Comparativa entre el sexo y la supervivencia

df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()

#Comparativa entre la puerta embarcada y la supervivencia

df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
#Procesamiento de data

df.Sex = df.Sex.map({'male':0, 'female':1})

df.Embarked = df.Embarked.map({'C': 1, 'Q': 2, 'S':3})

df.Embarked.fillna(4, inplace=True) #llena los que no tienen data de "embarked" con el valor 4
#Comparativa entre la cantidad de padres y la supervivencia

df.Parch = df.Parch.map(lambda x: 4 if x>4 else x)

df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean()
#Comparativa entre la cantidad de hermanos y la supervivencia

df.SibSp = df.SibSp.map(lambda x: 4 if x>4 else x)

df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean()
#Comparativa usando la mayoria de edad como parametro



df.Age = df.Age.map(lambda x: 0 if x<18 else 1)

df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean()
#Comparativa usando La clase en la que estaban abordo y la supervivencia

df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
#Se tratan los valores nulos

#Se procesa la data de la cabina

#Comparacion tipo de cabina y supervivencia

df.Cabin.fillna('Z', inplace=True)

df.Cabin = df.Cabin.map( lambda x: x[0])

df.Cabin = df.Cabin.map({'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8, 'Z':0})

df[['Cabin', 'Survived']].groupby(['Cabin'], as_index=False).mean()
#comparativa precio del boleto y supervivencia

df[['Fare', 'Survived']].groupby(['Survived'], as_index=False).mean()
#Regulizar valores con respecto al precio (Columna nueva)

#fare_regularizado = df.Fare.map(lambda x: 0 if x<35 else 1)

#pd.cut(fare_regularizado, labels=False, retbins=True)

#fare_regularizado
#Existen Nan en las edades por lo cual se busca llenar esos datos de la manera



#df['Title'] = df.Name.map( lambda x: x.split(',')[1].split( '.' )[0].strip())

#df['Title'] = df['Title'].replace('Mlle', 'Miss')

#df['Title'] = df['Title'].replace(['Lady','Mme','Ms'], 'Mrs')

#df['Title'] = df['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Sir', 'Don','Jonkheer','Dona', 'Capt', 'the Countess'], 'Other' )

#df['Title'].value_counts()
#borradera

#for i in range(0, len(df[['Title', 'Age']])):

    

  #  print( df.Age[i].value)

   # var = df.Age[i]! = df.Age[i]

  #  print(var)

   # if var is None:

    #    print(i)

      #  if df['Title'][i] is 'Master':

          #  df.Age[i] = 5 

      #  elif df['Title'][i] is 'Miss':

       #     df.Age[i] = 22 

      #  elif df['Title'][i] is 'Mr':

       #     df.Age[i] = 32 

       # elif df['Title'][i] is 'Mrs':

       #     df.Age[i] = 37 

        #elif df['Title'][i] is 'Other':

         #   df.Age[i] = 45 
df = df.drop(labels=['Name','Ticket','Age'], axis=1)

#valores de supervivencia de cada pasajero para el entrenamiento (valores esperados)

y_train = df[0:891]['Survived'].values

#data de entrenamiento

X_train = df[0:891].drop(['Survived','PassengerId'], axis=1).values 

#data de prueba

X_test  = df[891:].drop(['Survived','PassengerId'], axis=1).values

print("Train shape : ",y_train.shape)

print("Test shape : ",X_train.shape)
# Inicializando el NN

model = Sequential()



# layers

model.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 7))

model.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))

model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))

model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# Compilando la ANN

model.compile(optimizer = 'adam', loss = 'hinge', metrics = ['accuracy'])



# Train la ANN

model.fit(X_train, y_train, batch_size = 256, epochs = 350)
y_pred = model.predict(X_test)

y_final = (y_pred > 0.5).astype(int).reshape(X_test.shape[0])



output = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_final})

output.to_csv('prediction-ann.csv', index=False)