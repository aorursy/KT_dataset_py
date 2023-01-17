import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
# Load the dataset and create a DataFrame

# Se lee el dataset original como DataFrame



data_original = pd.read_csv("../input/titanic/titanic2.csv")





# A look to the datatypes and null values

# Se exploran los tipos de variables del dataset y los valores faltantes



#data_original.describe()

print(data_original.info())
# How the dataset looks like

# Cómo se ve el dataset



data_original.head()
# Setting up the new data types

dtypes_col       = data_original.columns

dtypes_type_old  = data_original.dtypes

dtypes_type      = ['int16', 'bool','category','object','category','float32','int8','int8','object','float32','object','category']

optimized_dtypes = dict(zip(dtypes_col, dtypes_type))



# Reading the entire data, but now with optimized columns

data = pd.read_csv("../input/titanic/titanic2.csv",dtype=optimized_dtypes)



print(data.info())
# Specifically checking null values

# Revisando valores faltantes



data.isnull().sum()
# Choice 1 (Easy): filling missing values with the Median of Age.

# Alternativa 1 (Básica): completar los faltantes con la mediana de Edad.



#data["Age"].fillna(data["Age"].median(), inplace = True)



# Choice 2 (Better): filling missing values taking the probability distribution of Age.

# Alternativa 2 (Más conveniente): completar los faltantes con valores tomados de la distribución de probabilidad de Age.



fig, ax = plt.subplots()

x = data["Age"].dropna()

hist, bins = np.histogram(x,bins=30)

ax.hist(x, bins= 30,color = 'blue')

ax.set_title('Histograma de Age')

ax.set_xlabel('Edad (años)')

ax.set_ylabel('Cantidad de datos / Passengers');
from random import choices



# Finding the probability of Age

bincenters = 0.5*(bins[1:]+bins[:-1])

probabilities = hist/hist.sum()



# Creating random numbers from existing age distribution

for item in data['Age']:

    data["Age_rand"] = data["Age"].apply(lambda v: np.random.choice(bincenters, p=probabilities))

    Age_null_list   = data[data["Age"].isnull()].index



    # Filling...   

    data.loc[Age_null_list,"Age"] = data.loc[Age_null_list,"Age_rand"]

    

data = data.drop(columns = ["Age_rand"])
# Checking null values

# Revisando valores faltantes

data.isnull().sum()
# Embarked could be filled with most popular category ("S"). Just becuase there are only 2 values missing.



# los 2 NaN de la columna Embarked se pueden reemplazar con S. Esto porque es la más popular y solo son 2 valores faltantes.



data["Embarked"] = data["Embarked"].fillna("S")

data.isnull().sum()
# Applying label encoding for category features (Sex, Embarked y Pclass) in order to be able of exploring correlations.



# Se hace label encoding para las tres variables que están como category (Sex, Embarked y Pclass). Esto para poder ver correlaciones.



# label encoding - variable sex

print('Valores originales en Sex:')

print(data["Sex"].unique())



from sklearn import preprocessing

le = preprocessing.LabelEncoder()



data["Sex"]=le.fit_transform(data["Sex"])

print('New values for Sex:')

print(data["Sex"].unique())
# label encoding - variable Embarked



print('Valores originales en Embarked:')

print(data["Embarked"].unique())



from sklearn import preprocessing

le = preprocessing.LabelEncoder()



data["Embarked"]=le.fit_transform(data["Embarked"])

print('New values for Embarked:')

print(data["Embarked"].unique())
# label encoding - variable Pclass



print('Valores originales en Pclass:')

print(data["Pclass"].unique())



from sklearn import preprocessing

le = preprocessing.LabelEncoder()



data["Pclass"]=le.fit_transform(data["Pclass"])

print('New values for Pclass:')

print(data["Pclass"].unique())
data.head()
# Cabin shouldn't be included in the Correlation Matrix due to the missing values.



# No se debe usar el campo Cabin porque hay muchos datos faltantes.

# El Passenger Id no creo que me sirva



corr = data.drop(columns = ['Cabin','PassengerId']).corr()

plt.figure(figsize=(8,8))

sns.heatmap(corr,linewidths=0.1,vmax=1.0, square=True, cmap="BuPu", linecolor='white', annot=True)

plt.title('Pearson correlation of numeric features', y=1.05, size=15)

plt.xticks(rotation = 45)

plt.yticks(rotation = 45)

plt.show()
# Sometimes it's worth to create groups of values. For instance, groups of Ages:

# A veces es útil crear agrupaciones de los valores de una variable. Por ejemplo Age:



#1: 0 a 12años

#2: 12 a 25

#3: 25 a 45

#4: >45



'''

fig, ax = plt.subplots()

plt.hist(data['Age'], bins=30, color='b')

ax.set(xlabel='Edades (años)', ylabel='Cantidad de datos / Passengers')

ax.set_title('Distribución de la edad de los pasajeros');



def formula(x):

    if x <= 12:

        return 1

    elif x > 12 and x <= 25:

        return 2

    elif x > 25 and x <= 45:

        return 3

    elif x > 45:

        return 4



data['age_group'] = data.apply(lambda row: formula(row['Age']), axis=1)



data['age_group'].value_counts()



'''
# 1. Creating features and targets

# 1. Definición de variables predictoras y variable a predecir



X = data[['Age','SibSp','Parch','Fare','Pclass','Sex','Embarked']]

y = data['Survived']



# 2. Splitting dataset into training data and validation data 

# 2. Separación de los datos en datos para entrenamiento y datos para validación



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# 3. Creating instance of the model

# 3. Instanciar el modelo



from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,

                       max_features=None, max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, presort=False,

                       random_state=None, splitter='best')



# 4. Training the model

# 4. Entrenar el modelo

tree.fit(X_train, y_train)



# 5. Predicting

# 5. Predecir



# Prediction using training data

y_train_pred = tree.predict(X_train)



# Predicción using validation data

y_test_pred = tree.predict(X_test)



# 6. Performance evaluation

from sklearn.metrics import accuracy_score



# Comparing with the real values

print('Accuracy sobre conjunto de Train:', accuracy_score(y_train_pred,y_train))

print('Accuracy sobre conjunto de Test:', accuracy_score(y_test_pred,y_test))
# Overfitting: cuando el error de entrenamiento es mayor al del testeo. Overfitting es alta varianza.



# El modelo solo tomará como válidos datos muy parecidos a los del conjunto de train.

# El modelo no reconocerá datos como buenos si está un poco por fuera de los rangos ya establecidos.
# Looking at the confusion matrix



from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(tree, X, y, cmap=plt.cm.Blues, values_format = '.2f', normalize= 'true')
# Understanding feature importances



importances = tree.feature_importances_

columns = X.columns

sns.barplot(columns, importances)

plt.title('Importancia de cada Feature')

plt.show()
# 1. Creating features and targets

# 1. Definición de variables predictoras y variable a predecir



X2 = data[['Sex','Fare','Age','Pclass']]

y2 = data['Survived']



# 2. Splitting dataset into training data and validation data 

# 2. Separación de los datos en datos para entrenamiento y datos para validación



from sklearn.model_selection import train_test_split

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42)



# 3. Creating instance of the model

# 3. Instanciar el modelo



tree2 = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,

                       max_features=None, max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, presort=False,

                       random_state=None, splitter='best')



# 4. Training the model

# 4. Entrenar el modelo



tree2.fit(X2_train, y2_train)





# 5. Predicting

# 5. Predecir



# Prediction using training data

y_train_pred2 = tree2.predict(X2_train)



# Prediction using validation data

y_test_pred2 = tree2.predict(X2_test)





# 6. Performance evaluation

from sklearn.metrics import accuracy_score



# Comparing with the real values

print('Accuracy sobre conjunto Train:', accuracy_score(y_train_pred2,y2_train))

print('Accuracy sobre conjunto Test:', accuracy_score(y_test_pred2,y2_test))
# Looking at the confusion matrix



from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(tree2, X2, y2, cmap=plt.cm.Blues, values_format = '.2f', normalize= 'true')
# Validation curve (from Scikit Learn):



# Defining a group of possible values for depth:

n = np.arange(2,50,2)



# Validation curve

from sklearn.model_selection import validation_curve



train_scores, test_scores = validation_curve(DecisionTreeClassifier(),

                                           X,

                                           y,

                                           param_name='max_depth',

                                           param_range=n,

                                             cv=5)



# Se revisan los parámetros de profundidad a evaluar:

#np.mean(train_scores,axis=1).shape



plt.plot(np.mean(train_scores,axis=1),color='darkblue',label='train_scores')

plt.plot(np.mean(test_scores,axis=1),color='darkgoldenrod',label='test_scores')

plt.xlabel('Profundidad del árbol')

plt.ylabel('Score')

plt.title('Curvas de validación - Decision Tree')

plt.xticks(np.arange(24),n);

plt.legend()

plt.show()
# Learning curve



from sklearn.model_selection import learning_curve



learning_curve(DecisionTreeClassifier(max_depth=16),X,y,cv=5)



lc = learning_curve(DecisionTreeClassifier(max_depth=16),X,y,cv=5)

samples, train, test = lc[0], lc[1], lc[2]



plt.plot(samples,np.mean(train,axis=1),color='dodgerblue',label='train_scores')

plt.plot(samples,np.mean(test,axis=1),color='orange',label='test_scores');

plt.xlabel('Datos')

plt.ylabel('Score')

plt.title('Curvas de aprendizaje - Árbol de decisión')

plt.legend()

plt.show()



print('Pareciera que al modelo ya le falta poco por aprender.')

print('La curva naranja (validation score) debería ser una asíntota. Pareciera que su pendiente ya no es muy alta.')
# 1. Creating features and targets

# 1. Definición de variables predictoras y variable a predecir



X3 = data[['Sex','Fare','Age','Pclass']]

y3 = data['Survived']



# 2. Splitting dataset into training data and validation data 

# 2. Separación de los datos en datos para entrenamiento y datos para validación



from sklearn.model_selection import train_test_split

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.3, random_state=42)



# 3. Creating instance of the model

# 3. Instanciar el modelo



from sklearn.tree import DecisionTreeClassifier

profundidad = 6

tree3 = DecisionTreeClassifier(max_depth=profundidad)



# 4. Training the model

# 4. Entrenar el modelo



tree3.fit(X3_train, y3_train)



# 5. Predicting

tree3_pred = tree3.predict(X3_test)

tree3_ajuste = tree3.predict(X3_train)



# 6. Performance evaluation

from sklearn.metrics import accuracy_score



print('DecTree optimizado (prof. ' + str(profundidad) + ')')

print('Accuracy = ' + str(round(accuracy_score(tree3_ajuste,y3_train),8)) + ' en conjunto Train.')

print('Accuracy = ' + str(round(accuracy_score(tree3_pred,y3_test),8)) + ' en conjunto Test.')