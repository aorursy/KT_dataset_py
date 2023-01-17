# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Configuración inicial y obtención de datos 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



df_train = pd.read_csv("/kaggle/input/titanic/train.csv", index_col = "PassengerId")

df_test  = pd.read_csv("/kaggle/input/titanic/test.csv", index_col = "PassengerId")



print("Los datos de entrenamiento contienen "+ str(df_train.shape[0])+" filas y "+str(df_train.shape[1])+" columnas >>")

print(df_train.columns)
df_train.drop(['Name', 'Ticket'], axis = 1, inplace  = True)

df_train.head()
# Resumen de los datos

print("Resumen:")

print(round(df_train[["Age","SibSp","Parch","Fare"]].describe().transpose(),1))



# Tipos de Variables 

print("Variables con Texto:")

for i in df_train.select_dtypes(include = ['object']):

    print("\t\t"+i)

print("Variables Numéricas:")

for i in df_train.select_dtypes(exclude = ['object']):

    print("\t\t"+i)



#Variables con valores nulos

print("Variables con Nulos")

missing_val_count = (df_train.isnull().sum())

print(missing_val_count[missing_val_count > 0])

# Primero revisamos qué categorías existen

var_cat = [ "Survived", "Pclass", 'Sex', 'Cabin', 'Embarked']

df_train[var_cat].astype('category')

df_train[var_cat].nunique()

df_train.Cabin.fillna("XX", inplace = True)

df_train['Cabin_letter'] =df_train.Cabin.str.slice(0, 1)

pd.crosstab([df_train.Cabin_letter],[df_train.Survived],

            margins=False, normalize='index').style.background_gradient(cmap='pink_r')

# Distribución por clase

var_cat = [ "Survived", "Pclass", 'Sex', 'Cabin_letter', 'Embarked']

for i in var_cat:

    print(i)

    c = df_train[i].value_counts(dropna=False)

    p = round(df_train[i].value_counts(dropna=False, normalize=True),1)

    print(pd.concat([c,p], axis=1, keys=['counts', '%']))
g = sns.catplot(x="Sex", y="Survived", hue="Pclass", col="Embarked", data=df_train, kind="bar",palette='pink');

g.set_xlabels("Sexo")

g.set_ylabels("Tasa de Supervivencia")

# Tabla cruzada para variables categóricas

pd.crosstab([df_train.Embarked,df_train.Pclass],[df_train.Sex,df_train.Survived],margins=True).style.background_gradient(cmap='pink_r')
# Análisis por sexo-edad-tarifa 

sns.relplot(x='Age', y='Fare', data=df_train,

            kind='scatter', hue='Survived', col='Sex', palette='pink_r')

# Análisis por clase 

sns.relplot(x='Age', y='Fare', data=df_train,

            kind='scatter', hue='Survived', col='Pclass', palette='pink_r')
# Pasaremos la edad a deciles para ver la dispersión más fácilemente con respecto a la tarifa del ticket

df_train["Age_q"] = pd.qcut(df_train['Age'], q=10, precision=0)

sns.catplot(

    data=df_train,

    x='Age_q',

    y='Fare',

    row='Survived',

    kind='box',

    height=3, 

    aspect=4,

    color='crimson')

# Finalmente revisado la distribbución y relación de las variables edad y tarifa

plt.figure(figsize=(20, 10))

sns.pairplot(df_train[["Age","Fare","Survived"]], kind="reg", diag_kind="kde", hue="Survived", palette="pink_r")

plt.show()

# Selección de datos

## Eliminanos la columna Cabin y nos quedamos sólo con la variable creada "Cabin_letter" 

df_train.drop("Cabin",axis = 1 ,  inplace = True)



# Completamos las dos filas con datos nulos en "Embarket" por la "Moda" = S 

df_train.loc[df_train["Embarked"].isnull() , "Embarked"] = "S"

#df_train.Embarked.value_counts()

# Y, la variable "Edad" reemplazamos por la media de edad de acuerdo a la variable supervivencia, por clase y sexo

for s in range(0, 2):

    for c in range(1, 4):

        for g in ["female","male"]:

            media = (df_train.Age[(df_train.Survived == s) & (df_train.Pclass == c) & (df_train.Sex == g)].mean())

            df_train.loc[(df_train.Age.isnull()) & (df_train.Survived == s) & (df_train.Pclass == c) & (df_train.Sex == g), "Age"] = media  

            df_test.loc[ (df_test.Age.isnull())  & (df_test.Pclass == c)  & (df_test.Sex == g) , "Age"] = media  



from sklearn.preprocessing import RobustScaler

# Revisamos los outliers

Q1 = df_train[["Age","Fare"]].quantile(0.25)

Q3 = df_train[["Age","Fare"]].quantile(0.75)

IQR = Q3 - Q1

print(IQR)

RS = RobustScaler(with_centering=False, with_scaling=True)

df_train[["Age","Fare"]] = RS.fit_transform(df_train[["Age","Fare"]])

# Las variables categóricas las transformaremos en columnas mediante la estrategia "One-Hot"

df_train_t = pd.get_dummies(df_train[[ "Survived","Age","Fare", 'Sex', 'Cabin_letter', 'Embarked']])

df_train_t = pd.concat( [df_train_t, pd.get_dummies(df_train.Pclass.astype(str))], axis = 1)

df_train_t.head()
from sklearn.model_selection import train_test_split

y = df_train_t["Survived"]

X = df_train_t.drop(['Survived'], axis=1, inplace=False)



X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)
from sklearn.neighbors import KNeighborsClassifier



modelo =  KNeighborsClassifier(n_neighbors=3)

modelo.fit(X_train, y_train)

from sklearn import metrics

# Model Accuracy, how often is the classifier correct?

y_pred = modelo.predict(X_valid)

print("Accuracy:",metrics.accuracy_score(y_valid, y_pred))
# Aplicamos las transformaciones al set de datos de test 

RS.transform(df_test[["Age","Fare"]])

df_test.loc[df_test["Embarked"].isnull() , "Embarked"] = "S"

df_test.loc[df_test["Fare"].isnull() , "Fare"] = df_train.Fare.mean() 

df_test.Cabin.fillna("XX", inplace = True)

df_test['Cabin_letter'] =df_test.Cabin.str.slice(0, 1)

df_test_t = pd.get_dummies(df_test[[ "Age","Fare", 'Sex', 'Cabin_letter', 'Embarked']])

df_test_t.insert(11, "Cabin_letter_T", 0) # No había nadie en la zona T 

df_test_t = pd.concat( [df_test_t, pd.get_dummies(df_test.Pclass.astype(str))], axis = 1)

df_test_t.head()
# Generate test predictions

preds_test = modelo.predict(df_test_t)





# Save predictions in format used for competition scoring

output = pd.DataFrame({'PassengerId': df_test_t.index,

                       'Survived': preds_test})

output.to_csv('submission_T_KNN_v2.csv', index=False)