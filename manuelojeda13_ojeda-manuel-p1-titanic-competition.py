import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline



# Ignorar warnings

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)



import keras

from keras.layers import Dense, Dropout

from keras.models import Input, Model



from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score



#Cargar Data

df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')

df = df_train.append(df_test , ignore_index = True)



#Inspecciones

df_train.shape, df_test.shape, df_train.columns.values
#Vericicar Nulls

df.Pclass.isnull().sum()
#Comparacion de las columnas Pclass y Survived

df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
#Grafica de comparacion de Pclass y Survived

sns.barplot(data=df,x='Pclass',y='Survived')
df['Title'] = df.Name.map( lambda x: x.split(',')[1].split( '.' )[0].strip())





#Contamos la cantida de personas que hay por "Titulo"

df['Title'].value_counts()
#Se modifican los valores en la tabla 

df['Title'] = df['Title'].replace('Mlle', 'Miss') #Se reemplaza Mlle por Miss

df['Title'] = df['Title'].replace(['Mme','Lady','Ms'], 'Mrs') #Se reemplaza Mme, Lady y Ms por Mrs

df.Title.loc[ (df.Title !=  'Master') & (df.Title !=  'Mr') & (df.Title !=  'Miss') #Todos los distintos se les asigna Others

             & (df.Title !=  'Mrs')] = 'Others'



# Comparando Title y Survived

df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

df = pd.concat([df, pd.get_dummies(df['Title'])], axis=1).drop(labels=['Name'], axis=1)
#Grafica Para analizar cuantas personas sobrevivieron segun su Title

sns.barplot(data=df,x='Title',y='Survived')
#Verificamos si hay Nulls 

df.Sex.isnull().sum()
#Comparando Sex y Survived

df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
#Grafica de comparacion de Sex y Survived

sns.barplot(data=df,x='Sex',y='Survived')
#Mapeamos la columna Sex 



df.Sex = df.Sex.map({'male':0, 'female':1})
#Verificamos si hay Nulls

df.Age.isnull().sum()
#Verificamos si hay null

df.SibSp.isnull().sum(axis=0), df.Parch.isnull().sum(axis=0)
#Creamos una columna nueva y Sumamos los valores en las columnas SibSp y Parch

df['Family'] = df['SibSp'] + df['Parch'] + 1
#Comparando Family y Survives

df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()
sns.barplot(data=df,x='Family',y='Survived')
#Contamos la cantidad de personas en familias

df['Family'].value_counts()
#Agrupamos las familias de 4 o mas integrantes

df.Family = df.Family.map(lambda x: 0 if x > 4 else x)
#Comparando Family y Survived

df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()
#Grafica de compracacion de Family y Survived

sns.barplot(data=df,x='Family',y='Survived')
#Verificamos si hay Nulls

df.Ticket.isnull().sum(axis=0)

#Traemos solo la primera letra de la columna ticket

df.Ticket = df.Ticket.map(lambda x: x[0])
# Comparando Ticket y Survived

df[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()
#Grafica de compracacion de Ticket y Survived

sns.barplot(data=df,x='Ticket',y='Survived')
#contamos la cantidad de personas por cada tipo de ticket

df['Ticket'].value_counts()

#Comparando Ticket con Fare

df[['Ticket', 'Fare']].groupby(['Ticket'], as_index=False).mean()

#Grafica de compracacion de Ticket y Fare

sns.barplot(data=df,x='Ticket',y='Fare')
#Comparando Ticket con Pclass

df[['Ticket', 'Pclass']].groupby(['Ticket'], as_index=False).mean()

#Grafica de compracacion de Ticket y Pclass

sns.barplot(data=df,x='Ticket',y='Pclass')
#Verificamos si hay nulls

df.Fare.isnull().sum()

#Buscandolo por el ticket

df.Ticket[df.Fare.isnull()]

#Buscandolo por la Pclass

df.Pclass[df.Fare.isnull()]

#Buscandolo por la cabina

df.Cabin[df.Fare.isnull()]

#Buscandolo por el puerto de embarque

df.Embarked[df.Fare.isnull()]

#Verificamos si hay Nulls

df.Cabin.isnull().sum(axis=0)
#Como hay tantos valores nulls Eliminamos la columna para no afectar los resultados

df = df.drop(labels=['Cabin'], axis=1)

# Verificamos si hay Nulls

df.Embarked.isnull().sum()
#Cambiamos las dos nulls asignandoles 'S'

df.Embarked.fillna('S',inplace=True)

#Grafica de comparacion entre Embarked y Survived

sns.barplot(data=df,x='Embarked',y='Survived')
#Comparando Embarked y Survived agregando otras caracteristicas

df[['Embarked', 'Survived','Pclass','Fare', 'Age', 'Sex']].groupby(['Embarked'], as_index=False).mean()
#Usando boxplot para visualizar la distribucion de la tarifa en cada Pclass



sns.boxplot('Pclass','Fare',data=df)

plt.ylim(0, 300) # ignorando la data cuya tarifa > 500

plt.show()
#Companando Pclass y Fare

df[['Pclass', 'Fare']].groupby(['Pclass']).mean()

#Grafica de comparacion entre Pclass y Fare

sns.barplot(data=df,x='Pclass',y='Fare')
#Dividimos la desviacion estandar por la media. Una relacion mas baja significa una distribucion mas apretada en la tarifa en cada clase

df[['Pclass', 'Fare']].groupby(['Pclass']).std() / df[['Pclass', 'Fare']].groupby(['Pclass']).mean()
#grafica de la distribucion de la tarifa por cada ticke

sns.boxplot('Ticket','Fare',data=df)

plt.ylim(0, 300) # ignorar la data con la tarifa mayor a 500

plt.show()
#Comparamos Ticket con Fare

df[['Ticket', 'Fare']].groupby(['Ticket']).mean()

#Se divide la desviasion estanar entre la media

df[['Ticket', 'Fare']].groupby(['Ticket']).std() /  df[['Ticket', 'Fare']].groupby(['Ticket']).mean()
#Grafica de Tarifas por cada puerto de embarque

sns.boxplot('Embarked','Fare',data=df)

plt.ylim(0, 300) # ignoramos la data cuya tarifa sea mayor que 500

plt.show()
#Comparando Embarked y Fare

df[['Embarked', 'Fare']].groupby(['Embarked']).mean()

#Grafica de comparacion de Embarked y Fare

sns.barplot(data=df,x='Embarked',y='Fare')
#Se divide la desviasion estandar entre la media

df[['Embarked', 'Fare']].groupby(['Embarked']).std() /  df[['Embarked', 'Fare']].groupby(['Embarked']).mean()

guess_Fare = df.Fare.loc[ (df.Ticket == '3') & (df.Pclass == 3) & (df.Embarked == 'S')].median()

df.Fare.fillna(guess_Fare , inplace=True)



# Vemos los valores de la media de la Tarifa para las personas que murieron y sobrevivieron

df[['Fare', 'Survived']].groupby(['Survived'],as_index=False).mean()
#Visualizamos la distribucion de la tarifa de las personas que sobrevivieron y de las que murieron



grid = sns.FacetGrid(df, hue='Survived', height=4, aspect=1.5)

grid.map(plt.hist, 'Fare', alpha=.5, bins=range(0,210,10))

grid.add_legend()

plt.show()
#Vemos la relacion entre la Tarifa y Survided usando un scatter plot



df[['Fare', 'Survived']].groupby(['Fare'],as_index=False).mean().plot.scatter('Fare','Survived')

plt.show()
#Dividimos la Tarifa en 5 intervalos con la misma cantidad de personas (Fare-Bin)

df['Fare-bin'] = pd.qcut(df.Fare,5,labels=[1,2,3,4,5]).astype(int)



# Comparamos Fare-bin y Survived

df[['Fare-bin', 'Survived']].groupby(['Fare-bin'], as_index=False).mean()
#Comparamos entre Title y Age

grid = sns.FacetGrid(df, col='Title', height=3, aspect=0.8, sharey=False)

grid.map(plt.hist, 'Age', alpha=.5, bins=range(0,105,5))

plt.show()
#Visuaizamos la media de Age por cada Title

df[['Title', 'Age']].groupby(['Title']).mean()

#Visuaizamos la desviasion estandar de Age por cada Title

df[['Title', 'Age']].groupby(['Title']).std()

#Comparamos Fare-bin y Age



grid = sns.FacetGrid(df, col='Fare-bin', height=3, aspect=0.8, sharey=False)

grid.map(plt.hist, 'Age', alpha=.5, bins=range(0,105,5))

plt.show()
#Comparacion entre SibSp y Age

grid = sns.FacetGrid(df, col='SibSp', col_wrap=4, height=3.0, aspect=0.8, sharey=False)

grid.map(plt.hist, 'Age', alpha=.5, bins=range(0,105,5))

plt.show()
#Visuaizamos la media de Age por cada SibSp

df[['SibSp', 'Age']].groupby(['SibSp']).mean()

#Visuaizamos la desviasion estandar de Age por cada SibSp



df[['SibSp', 'Age']].groupby(['SibSp']).std()

#Comparacion entre Parch y Age



grid = sns.FacetGrid(df, col='Parch', col_wrap=4, height=3.0, aspect=0.8, sharey=False)

grid.map(plt.hist, 'Age', alpha=.5, bins=range(0,105,5))

plt.show()
#Visuaizamos la media de Age por cada Parch



df[['Parch', 'Age']].groupby(['Parch']).mean()

#Visualizamos la desviacion estandar de Age por cada Parch



df[['Parch', 'Age']].groupby(['Parch']).std()

#Eliminamos la columna de embarque por ofrecer poca informacion relevante

df = df.drop(labels='Embarked', axis=1)
#Notamos que en lugar de utilizar Title, deberiamos usar su correspondiente dummy variables







df_sub = df[['Age','Master','Miss','Mr','Mrs','Others','Fare-bin','SibSp']]



X_train  = df_sub.dropna().drop('Age', axis=1)

y_train  = df['Age'].dropna()

X_test = df_sub.loc[np.isnan(df.Age)].drop('Age', axis=1)



regressor = RandomForestRegressor(n_estimators = 300)

regressor.fit(X_train, y_train)

y_pred = np.round(regressor.predict(X_test),1)

df.Age.loc[df.Age.isnull()] = y_pred



df.Age.isnull().sum(axis=0) # ya no mas null
bins = [ 0, 4, 12, 18, 30, 50, 65, 100] # Esto es algo arbitrario

age_index = (1,2,3,4,5,6,7) #('baby','child','teenager','young','mid-age','over-50','senior')

df['Age-bin'] = pd.cut(df.Age, bins, labels=age_index).astype(int)



df[['Age-bin', 'Survived']].groupby(['Age-bin'],as_index=False).mean()
df[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()

df['Ticket'].value_counts()

df['Ticket'] = df['Ticket'].replace(['A','W','F','L','5','6','7','8','9'], '4')



# Comparamos nuevamente

df[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()
df = pd.get_dummies(df,columns=['Ticket'])

df = df.drop(labels=['SibSp','Parch','Age','Fare','Title'], axis=1)

y_train = df[0:891]['Survived'].values

X_train = df[0:891].drop(['Survived','PassengerId'], axis=1).values

X_test  = df[891:].drop(['Survived','PassengerId'], axis=1).values
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
# Normalizamos los inputs

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
print("Entrenamiento: ",X_train.shape)

print("Test : ",X_test.shape)
X_train,val_dfX,y_train, val_dfY = train_test_split(X_train,y_train , test_size=0.20, stratify=y_train)

print("Entrenamiento: ",X_train.shape)

print("Validacion : ",val_dfX.shape)
def func_model():   

    inp = Input(shape=(17,))

    x=Dense(8, activation="relu", kernel_initializer='glorot_normal')(inp)

    x=Dense(8, activation="relu", kernel_initializer='glorot_normal')(x)

    x=Dense(1, activation="sigmoid", kernel_initializer='glorot_normal')(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['binary_accuracy'])

    return model

model = func_model()

print(model.summary())
train_history = model.fit(X_train, y_train, batch_size = 32, epochs = epochs, validation_data=(val_dfX, val_dfY))
graf_model(train_history)

precision(model, True)
def func_model_reg():   

    inp = Input(shape=(17,))

    x=Dropout(0.1)(inp)

    x=Dense(8, activation="relu", kernel_initializer='glorot_normal')(inp)

    x=Dropout(0.3)(x)

    x=Dense(8, activation="relu", kernel_initializer='glorot_normal')(x)

    x=Dropout(0.3)(x)

    x=Dense(1, activation="sigmoid", kernel_initializer='glorot_normal')(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['binary_accuracy'])

    return model
modelR = func_model_reg()

print(modelR.summary())

finalTrain = modelR.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(val_dfX, val_dfY), verbose=0)
graf_model(finalTrain)

precision(modelR)
y_pred = model.predict(X_test)

y_final = (y_pred > 0.5).astype(int).reshape(X_test.shape[0])



output = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_final})

output.to_csv('prediction.csv', index=False)