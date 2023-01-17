import pandas as pd

import numpy as np

import random as rd



import seaborn as sns

import matplotlib.pyplot as plt

import pylab as plot



params = { 

    'axes.labelsize': "large",

    'xtick.labelsize': 'x-large',

    'legend.fontsize': 20,

    'figure.dpi': 150,

    'figure.figsize': [25, 7]

}

plot.rcParams.update(params)


train_titanic= pd.read_csv ('../input/train.csv')

test_titanic= pd.read_csv ('../input/test.csv')

print (train_titanic.shape)

print (train_titanic.columns.values)
train_titanic.info()
train_titanic.describe()
#pandas.DataFrame.fillna

#Utilizada para reemplazar los valores NA/NaN con un método especificado

#Referencia: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html

train_titanic['Age']=train_titanic['Age'].fillna(train_titanic['Age'].median())
#En la función seaborn, el parámetro "hue" determina qué columna en el 

#dataframe debe ser utilizada para codificar los colores.

#Referencia:

#https://seaborn.pydata.org/generated/seaborn.pointplot.html

fig=plt.figure(figsize=(20,10))

sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=train_titanic)

fig=plt.figure(figsize=(20,10))

sns.violinplot(x="Sex", y="Age", hue="Survived", data=train_titanic, split=True, palette={0: "b", 1: "g"});
fig=plt.figure(figsize=(20,10))

sns.barplot(x="Embarked", y="Survived", hue="Sex", data=train_titanic)

fig=plt.figure(figsize=(20,10))

sns.violinplot(x="Embarked", y="Fare", hue="Survived", data=train_titanic, split=True, palette={0: "b", 1: "g"});
figure = plt.figure(figsize=(20, 10))

plt.hist([train_titanic[train_titanic['Survived'] == 1]['Fare'], train_titanic[train_titanic['Survived'] == 0]['Fare']], 

         stacked=False, color = ['g','b'],

         bins = 50, label = ['Survived','Dead'])

plt.xlabel('Fare')

plt.ylabel('Number of passengers')

plt.legend();
#Leyendo los datos de los archivos .csv

training_data=pd.read_csv('../input/train.csv')

testing_data=pd.read_csv('../input/test.csv')



#La variable "Survived" no se encuentra en el dataset de "test",

#por lo cuál, será eliminada.



#Primero guardaremos el valor de la variable para no perder los datos

Survived=training_data['Survived']

#Validando

Survived.head()

#Vamos a eliminar la variable "Survived" permanentemente

#Para eso, utilizamos el campo inplace="True", de otro modo sólo la eliminamos

#de la vista, pero no permanéntemente

#Referencia: https://www.ritchieng.com/pandas-inplace-parameter/

training_data.drop('Survived', axis=1, inplace=True)

training_data.head()

#Obteniendo las dimensiones de los dos datasets, ten en cuenta que ya eliminamos

#la variable "Survived", así que el número de campos debe de estar homologado

#(Reflejado en el número de columnas)

print (training_data.shape)

print (testing_data.shape)

complete_dataset=training_data.append(testing_data)

#Comprobando que el número de columnas efectívamente corresponde a la unión

#de los dos datasets

print(complete_dataset.shape)
complete_dataset.drop('PassengerId',axis=1,inplace=True)

#Validando que el campo ya no existe

complete_dataset.head()

print (complete_dataset.shape)
complete_dataset.head()
titles=set()

#Realizando una iteración en los datos para separarlos.

for name in complete_dataset['Name']:

#Al hacer referencia al índice 1, hacemos referencia a la segunda parte del string

#que es donde está el título y el resto del nombre.

#El título va separado por un punto, así que podemos hacer una segunda función

#que separe, utilizando el punto, y así obtener el título

#La función de strip símplemente elimina los espacios en blanco

#Referencia:

#https://www.programiz.com/python-programming/methods/string/strip



    #print (name.split(',')[0])

    titles.add(name.split(',')[1].split('.')[0].strip())

print (titles)
Title_Dictionary = {

                    "Capt":       "Officer",

                    "Col":        "Officer",

                    "Major":      "Officer",

                    "Jonkheer":   "Royalty",

                    "Don":        "Royalty",

                    "Sir" :       "Royalty",

                    "Dr":         "Officer",

                    "Rev":        "Officer",

                    "the Countess":"Royalty",

                    "Dona":       "Royalty",

                    "Mme":        "Mrs",

                    "Mlle":       "Miss",

                    "Ms":         "Mrs",

                    "Mr" :        "Mr",

                    "Mrs" :       "Mrs",

                    "Miss" :      "Miss",

                    "Master" :    "Master",

                    "Lady" :      "Royalty"



                    }
complete_dataset['Title']=complete_dataset['Name'].map(lambda name: name.split( ',' )[1].split( '.' )[0].strip())

complete_dataset['Title']=complete_dataset.Title.map(Title_Dictionary)

complete_dataset.tail()

complete_dataset.info()
#La función value_counts nos ayuda a determinar la frecuencia de cada una de

#nuestras variables en el campo "Embarked"

complete_dataset['Embarked'].value_counts()
#Reemplazando las variables con valores faltantes:

complete_dataset['Embarked']=complete_dataset['Embarked'].fillna('S')

#train_titanic['Age']=train_titanic['Age'].fillna(train_titanic['Age'].median())
complete_dataset.info()

complete_dataset.head()
complete_dataset['Fare']=complete_dataset['Fare'].fillna(complete_dataset['Fare'].median())

complete_dataset.info()

complete_dataset.drop('Name', axis=1, inplace=True)
complete_dataset.info()
complete_dataset['Age']=complete_dataset['Age'].fillna(complete_dataset['Age'].median())
complete_dataset.info()
complete_dataset['Cabin']=complete_dataset['Cabin'].fillna('U')
complete_dataset.info()
complete_dataset['Cabin'] = complete_dataset['Cabin'].map(lambda c: c[0])
complete_dataset['Cabin'].value_counts()
complete_dataset['Family']=complete_dataset['SibSp']+complete_dataset['Parch']+1

#El número 1 es tomando en cuenta al pasajero, símplemente estamos sumando

#el número de acompañantes con el pasajero
#Podemos definir el tamaño de una familia de acuerdo al resultado del nuevo campo

complete_dataset['OneMemberFamily']=complete_dataset['Family'].map(lambda s: 1 if s==1 else 0) 

complete_dataset['SmallFamily']=complete_dataset['Family'].map(lambda s: 1 if s>= 2 and s<=4 else 0)

complete_dataset['BigFamily']=complete_dataset['Family'].map(lambda s: 1 if s>4 else 0)
complete_dataset.head()
complete_dataset['Ticket']=complete_dataset['Ticket'].str.replace('.','')

complete_dataset['Ticket']=complete_dataset['Ticket'].str.replace('/','')

complete_dataset['Ticket']=complete_dataset['Ticket'].str.replace('\d+','')

#complete_dataset['Ticket']=complete_dataset['Ticket'].map(lambda x: x.lstrip('/').rstrip(''))
complete_dataset.info()
complete_dataset.head(20)
complete_dataset['Ticket']=complete_dataset['Ticket'].map(lambda name: name.strip())

complete_dataset['Ticket']=complete_dataset['Ticket'].map(lambda name: "XXX" if len(name)==0 else name)
complete_dataset.head(20)
#Creando nuestras variables dummy

dummy_title=pd.get_dummies(complete_dataset['Title'])

dummy_title.info()
complete_dataset=pd.concat([complete_dataset, dummy_title], axis=1)
complete_dataset.head()
complete_dataset.drop('Title', axis=1, inplace=True)
complete_dataset.head()
#Procesamiento de la variable "Embarked"

dummy_embarked=pd.get_dummies(complete_dataset['Embarked'], prefix='Embarked')

complete_dataset=pd.concat([complete_dataset, dummy_embarked], axis=1)

complete_dataset.drop('Embarked', axis=1, inplace=True)
#Procesamiento de la variable "Cabin"

dummy_cabin=pd.get_dummies(complete_dataset['Cabin'], prefix='Cabin')

complete_dataset=pd.concat([complete_dataset, dummy_cabin], axis=1)

complete_dataset.drop('Cabin', axis=1, inplace=True)
#Procesamiento de la variable "Pclass"

dummy_pclass=pd.get_dummies(complete_dataset['Pclass'], prefix='Pclass')

complete_dataset=pd.concat([complete_dataset, dummy_pclass], axis=1)

complete_dataset.drop('Pclass', axis=1, inplace=True)
#Procesamiento de la variable "Sex"

complete_dataset['Sex']=complete_dataset['Sex'].map({'male': 1, 'female':0 })

#dummy_sex=pd.get_dummies(complete_dataset['Sex'])

#complete_dataset=pd.concat([complete_dataset, dummy_sex], axis=1)

#complete_dataset.drop('Sex', axis=1, inplace=True)
#Procesamiento de la variable "Ticket"

dummy_ticket=pd.get_dummies(complete_dataset['Ticket'], prefix='Ticket')

complete_dataset=pd.concat([complete_dataset, dummy_ticket], axis=1)

complete_dataset.drop('Ticket', axis=1, inplace=True)
## complete_dataset.head()
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
Train_Titanic_customized=complete_dataset[:891]

Test_Titanic_customized=complete_dataset[891:]

#Necesitamos obtener nuestras variables que reprsentarán el ground truth

targets=pd.read_csv('../input/train.csv',usecols=['Survived'])['Survived'].values

classifier = RandomForestClassifier(n_estimators=50, max_features='sqrt')

classifier = classifier.fit(Train_Titanic_customized, targets)
features = pd.DataFrame()

features['feature'] = Train_Titanic_customized.columns

features['importance'] = classifier.feature_importances_

features.sort_values(by=['importance'], ascending=True, inplace=True)

features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(30, 20))
#Reduciendo el número de nuestras variables:

model = SelectFromModel(classifier, prefit=True)

Train_Titanic_reduced = model.transform(Train_Titanic_customized)

print (Train_Titanic_reduced.shape)
Test_Titanic_reduced = model.transform(Test_Titanic_customized)

print (Test_Titanic_reduced.shape)
logreg = LogisticRegression()

logreg_cv = LogisticRegressionCV()

rf = RandomForestClassifier()

gboost = GradientBoostingClassifier()



models = [logreg, logreg_cv, rf, gboost]
for model in models:

    print ('Cross-validation of: {0}'.format(model))

    score = cross_val_score(classifier, Train_Titanic_reduced, targets, cv = 5, scoring='accuracy')

    print ('CV score = {0}'.format(score))

    print ('****')
#Generando un modelo que nos de los outputs para predecir el índice de supervivencia en el dataset



parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 

                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6} 



model = RandomForestClassifier(**parameters)

model.fit(Train_Titanic_reduced, targets)

output = model.predict(Test_Titanic_reduced).astype(int)

df_output = pd.DataFrame()

aux = pd.read_csv('../input/test.csv')

df_output['PassengerId'] = aux['PassengerId']

df_output['Survived'] = output

df_output[['PassengerId','Survived']].to_csv('../input/Daniel_predictions.csv', index=False)