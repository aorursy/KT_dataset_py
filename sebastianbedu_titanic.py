#importar librerías

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

#from plotnine import *





#sección del tutorial para listar la ruta de los datos de entrenamiento y a predecir

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

#Notas del tutorial

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Leer dataset de entrenamiento

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
#Leer dataset a predecir

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
sns.set()

sns.catplot(x="Sex", hue="Pclass", kind="count", legend=False,data=train_data)

plt.title('Total por género y clase')

plt.xlabel('Género')

plt.ylabel('No. Tripulantes')

plt.legend(title='Clase', loc='upper right', labels=['1ra', '2da', '3ra'])
sns.catplot(x="Sex", hue="Survived", kind="count", legend=False,data=train_data)

plt.title('Total sobrevivientes por género')

plt.xlabel('Género')

plt.ylabel('No. Tripulantes')

plt.legend(title='Status', loc='upper right', labels=['No Sobrevivió', 'Sobrevivió'])
#Porcentaje de mujeres que sobrevivieron

women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% de mujeres que sobrevivieron:", rate_women)
#Porcentaje de hombres que sobrevivieron

men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% de hombres que sobrevivieron:", rate_men)
print('Total hombres:',len(men))

print('Total mujeres:',len(women))

print('Total sobrevivientes hombres:',sum(men))

print('Total sobrevivientes mujeres:',sum(women))
#Parch = 0 que sobrevivieron

parch_0 = train_data.loc[train_data.Parch == 0]["Survived"]

rate_parch_0 = sum(parch_0)/len(parch_0)

print("% parch = 0 que sobrevivieron:", rate_parch_0)



parch_1 = train_data.loc[train_data.Parch == 1]["Survived"]

rate_parch_1 = sum(parch_1)/len(parch_1)

print("% parch = 1 que sobrevivieron:", rate_parch_1)



parch_2 = train_data.loc[train_data.Parch == 2]["Survived"]

rate_parch_2 = sum(parch_2)/len(parch_2)

print("% parch = 2 que sobrevivieron:", rate_parch_2)



parch_3 = train_data.loc[train_data.Parch == 3]["Survived"]

rate_parch_3 = sum(parch_3)/len(parch_3)

print("% parch = 3 que sobrevivieron:", rate_parch_3)



parch_4 = train_data.loc[train_data.Parch == 4]["Survived"]

rate_parch_4 = sum(parch_4)/len(parch_4)

print("% parch = 4 que sobrevivieron:", rate_parch_4)



parch_5 = train_data.loc[train_data.Parch == 5]["Survived"]

rate_parch_5 = sum(parch_5)/len(parch_5)

print("% parch = 5 que sobrevivieron:", rate_parch_5)



parch_6 = train_data.loc[train_data.Parch == 6]["Survived"]

rate_parch_6 = sum(parch_6)/len(parch_6)

print("% parch = 3 que sobrevivieron:", rate_parch_6)





print('----')

print('Parch0 sobrevivientes sobre el total',sum(parch_0)/len(train_data['Parch']))

print('Parch1 sobrevivientes sobre el total',sum(parch_1)/len(train_data['Parch']))

print('Parch2 sobrevivientes sobre el total',sum(parch_2)/len(train_data['Parch']))

print('Parch3 sobrevivientes sobre el total',sum(parch_3)/len(train_data['Parch']))

print('Parch4 sobrevivientes sobre el total',sum(parch_4)/len(train_data['Parch']))

print('Parch5 sobrevivientes sobre el total',sum(parch_5)/len(train_data['Parch']))

print('Parch6 sobrevivientes sobre el total',sum(parch_6)/len(train_data['Parch']))







print(sum(train_data['Survived'])/len(train_data['Survived']))
print("Total de datos usando survived:",len(train_data['Survived']))



print("Total de datos usando sex:",len(train_data['Sex']))



print("Total hombres:",len(men))

print("Total mujeres:",len(women))

print("Total hombres vivos:", sum(men))

print("Total mujeres vivas:",sum(women))



print(train_data.describe())

#Faltan datos dela edad

print(test_data.describe())
#Revisar columnas que tengan null(vacíos)

print('Vacios en embarked:',train_data['Embarked'].isnull().values.any())

print('Vacios en Sex:',train_data['Sex'].isnull().values.any())

print('Vacios en Name:',train_data['Name'].isnull().values.any())

print('Vacios en Age:',train_data['Age'].isnull().values.any())

print('Vacios en Cabin:',train_data['Cabin'].isnull().values.any())

print('Vacios en Ticket:',train_data['Ticket'].isnull().values.any())



#Vacios

#embarked, Age y Cabin
#Ver cuantos registros están vacios

print('Embarked:',train_data['Embarked'].isnull().sum())

print('Age:',train_data['Age'].isnull().sum())

print('Cabin:',train_data['Cabin'].isnull().sum())
#Que tipo de datos se tienen?



print('Embarked:',train_data['Embarked'].unique())

print('Sex:',train_data['Sex'].unique())

print('Parch:',train_data['Parch'].unique())

embarked_s = train_data.loc[train_data.Embarked == 'S']["Survived"]

rate_embarked_s = sum(embarked_s)/len(train_data['Survived'])

print('% que sobrevivio con embarked = S: ',rate_embarked_s)



embarked_c = train_data.loc[train_data.Embarked == 'C']["Survived"]

rate_embarked_c = sum(embarked_c)/len(train_data['Survived'])

print('% que sobrevivio con embarked = C:',rate_embarked_c)



embarked_q = train_data.loc[train_data.Embarked == 'Q']["Survived"]

rate_embarked_q = sum(embarked_q)/len(train_data['Survived'])

print('% que sobrevivio con embarked = Q:',rate_embarked_q)



print('s:',len(embarked_s), 'c:',len(embarked_c), 'q:',len(embarked_q))



#Sobrevivieron más de los que embarcaron en S pero erán más



embarked_s = train_data.loc[train_data.Embarked == 'S']["Survived"]

rate_embarked_s = sum(embarked_s)/len(embarked_s)

print('%S: ',rate_embarked_s)



embarked_c = train_data.loc[train_data.Embarked == 'C']["Survived"]

rate_embarked_c = sum(embarked_c)/len(embarked_c)

print('%C:',rate_embarked_c)



embarked_q = train_data.loc[train_data.Embarked == 'Q']["Survived"]

rate_embarked_q = sum(embarked_q)/len(embarked_q)

print('%Q:',rate_embarked_q)


train_data[train_data['Embarked'].isnull()]



#Porque tienen el mismo ticket id???? R= https://www.encyclopedia-titanica.org/titanic-survivor/amelia-icard.html

#Con el dato de la info se sabe que embarked es = S
train_data[train_data['Age'].isnull()]

#TODO: Sacar el rango de Fare y compararlo con los que si tienen el Cabin
#train_data[train_data['Pclass'] == 1]



primera_clase = train_data.loc[train_data.Pclass == 1]["Survived"]

rate_1 = sum(primera_clase)/len(primera_clase)



print("% Sobrevivientes en primera clase:", rate_1)



segunda_clase = train_data.loc[train_data.Pclass == 2]["Survived"]

rate_2 = sum(segunda_clase)/len(segunda_clase)



print("% Sobrevivientes en segunda clase:", rate_2)



tercera_clase = train_data.loc[train_data.Pclass == 3]["Survived"]

rate_3 = sum(tercera_clase)/len(tercera_clase)



print("% Sobrevivientes en tercera clase:", rate_3)



#De la primera clase hubo más sobrevivientes
test_data.head(10)
#Ya que se tienen solo 2 valores de la columna embarked en NaN pueden borrarse esas filas o poner el valor más repetido de esa columna

#Ya que se sabe que subieron en S se setteara dicho valor

train_data['Embarked'].fillna('S', inplace=True)

#Comprobar el cambio

train_data[(train_data['PassengerId'] == 62) | (train_data['PassengerId'] == 830)]
#Algunos registros tienen cabin, la mayoría no, se podra tomar como 1 y 0 ? para usarlo en el modelo...



    

#train_data['Cabin'] = np.where(train_data['Cabin'].isnull(), 0, 1)





#Se tiene que hacer lo mismo para los datos a predecir para que funcione el modelo

#test_data['Cabin'] = np.where(test_data['Cabin'].isnull(), 0, 1)





#Trabajar con cabin





train_data['Cabin'].fillna('M', inplace=True)

train_data['Cabin_full'] = train_data.Cabin.str.slice(0, 1)



test_data['Cabin'].fillna('M', inplace=True)

test_data['Cabin_full'] = test_data.Cabin.str.slice(0, 1)
train_data.head(10)
test_data.head(10)
#Ya no hay ningún dato vacio de cabin, se podría utilizar en el modelo 



train_data[train_data['Cabin'].isnull()]
#Trabajar con la edad

#https://medium.com/vickdata/four-feature-types-and-how-to-transform-them-for-machine-learning-8693e1c24e80

#Llenar edad con la mediana

train_data["Age"].fillna(train_data["Age"].mean(), inplace=True)

#train_data.describe()





#Poner el promedio tomando en cuenta subconjuntos (Pclass y Sex)

#Por ejemplo: Promedio de los datos que sean Pclass = 1 y Sex =  female 

#Visto en: https://www.udemy.com/course-dashboard-redirect/?course_id=3034216 

#Se toma el promedio  de todos los datos o separados train y test?



#train_data.groupby(["Sex","Pclass"]).agg({'Age': ['mean']})





train_data["Age"].max(),test_data["Age"].max()
#Transformar edad en rangos (categorías)

train_data['age_bins'] = pd.cut(x=train_data['Age'], bins=[0, 12, 20, 25, 40, 80])

train_data['age_bins'].unique()
train_data.head()
#Mismo caso para test_data

test_data["Age"].fillna(test_data["Age"].mean(), inplace=True)

test_data.describe()
test_data['age_bins'] = pd.cut(x=test_data['Age'], bins=[0, 12, 20, 25, 40, 80])

test_data['age_bins'].unique()
test_data.head()
#Transformar Pclass a categorica esto dio de resultado 0.77033 en el submit

#train_data['Pclass'] = pd.Categorical(train_data.Pclass)

#test_data['Pclass'] = pd.Categorical(test_data.Pclass)

train_data.corr(method ='pearson') 
sns.catplot(x="age_bins", hue="Survived", kind="count", legend=False,data=train_data)

plt.title('Total sobrevivientes por rango de edad')

plt.xlabel('Edad')

plt.ylabel('No. Tripulantes')

plt.legend(title='Status', loc='upper right', labels=['No Sobrevivió', 'Sobrevivió'])
#Saber si estuvieron solos o no





#Para train data

train_data['vaSolo'] = train_data['SibSp'] + train_data['Parch']

train_data['vaSolo'] = np.where(train_data['vaSolo'] == 0, 1, 0)



train_data.head()


#Para test data

test_data['vaSolo'] = test_data['SibSp'] + test_data['Parch']

test_data['vaSolo'] = np.where(test_data['vaSolo'] == 0, 1, 0)



train_data.head()
train_data.head()
test_data.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



#Definir variables predictoras (X) y variable objetivo/respuesta/dependiente (y)

#Transformar de texto a enteros con getdummies, generando columna hombre y mujer 

#Ya que la clase, el género y donde embarcaron parece influir se tomarán como los features

#Ya se puede usar cabin

features = ['Pclass', 'Sex', 'Embarked', 'age_bins', 'vaSolo', 'Cabin_full']



y = train_data['Survived'].values

X = pd.get_dummies(train_data[features])



#Del dataset de entrenamiento dividir para tener datos para entrenar y para predecir, así poder ver el accuracy

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.40, random_state=5, stratify=y)









X_test_r = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=300, max_depth=4, random_state=0, max_features=4)

model.fit(X_train, y_train)



y_pred= model.predict(X_test)



model.score(X_test, y_test)
X_test_r.head()

#No se tiene la columna T, se agrega con valores en 0

X_test_r['Cabin_full_T'] = 0
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

#print(model.score(X_test, y_test))



#La matriz de confusión es mejor para  ver que tan acertado es el modelo

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

for feature in zip(features, model.feature_importances_):

    print(feature)

predictions = model.predict(X_test_r)


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Se guardó el csv")

#mejoró el porcentaje pero no superó al de la versión 4