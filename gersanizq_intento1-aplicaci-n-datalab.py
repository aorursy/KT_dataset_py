#Importe muchas cosas, esto lo copié de alguna página. 

import pandas as pd

import numpy as np

import random as rnd

from pprint import pprint as pp



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



from sklearn import metrics



#matplotlib

import matplotlib.pyplot as plt
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
len(train_df)
len(test_df)
train_df.info()
train_df.describe()
def label_Sex (row):

    if row['Sex'] == "male" :

        return 0

    return 1
train_df['SexNum'] = train_df.apply(lambda row: label_Sex(row),axis=1)
train_df[['Sex','SexNum']].head()
def label_Port (row):

    if row['Embarked'] == "C" :

        return 0

    if row['Embarked'] == "Q":

        return 1

    return 2
train_df['EmbarkedNum'] = train_df.apply(lambda row: label_Port(row),axis=1)
train_df[['Embarked', 'EmbarkedNum']].head(20)
def label_Fare (row):

    if row['Fare'] > 0 :

        return row['Fare']

    return 32 #Si es NaN, le asigno el fare promedio
#La función que sigue categoriza a las personas de acuerdo a su edad: 

def label_Age (row):

    if row['Age'] < 10 :

        return 1

    if row['Age'] < 15:

        return 2

    if row['Age'] <55:

        return 3

    if row['Age']<110:

        return 4

    return 0
#def label_Age (row):

 #   if row['Age'] < 110 :

  #      return row['Age']

   # return 29.6

    

    #Aquí intente sustituir los NaN's por el promedio, pero bajaba la precisión de la predicción.

    #Intente también hacer una regresión que estime la edad para los NaN's para luego correr los modelos sobre las edades

    #con las edades estimadas remplazando a los NaN's, pero aún no soy tan bueno en Python. 
train_df['AgeNum'] = train_df.apply(lambda row: label_Age(row),axis=1)
train_df[['Age', 'AgeNum']].head(20)
#Esto lo copié de varios lugares de internet.



dfAux = train_df[['Pclass', "SexNum", 'SibSp', 'Parch', 'Fare', 'EmbarkedNum', 'AgeNum']]

dfAux2 = train_df['Survived']



matrizPropiedades = dfAux.values

vectorObjetivo = dfAux2.values



pp(matrizPropiedades)

pp(vectorObjetivo)
datos = np.array(matrizPropiedades)

target = np.array(vectorObjetivo)

print(target)
len(datos)
len(target)
#Aquí separo los datos train y test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(datos, target, test_size=0.45, random_state=42)
#Utilizo este (GaussianNB) porque fue el que mejor me funcionó para la proporción 0.45, que elegí por parecerse a la proporción real.

model = GaussianNB()

model.fit(X_train, y_train)

predicted = model.predict(X_test)



# summarize the fit of the model

print(metrics.accuracy_score(y_test, predicted))

print()

print(metrics.classification_report(y_test, predicted))

print(metrics.confusion_matrix(y_test, predicted))

print()
test_df['AgeNum'] = test_df.apply(lambda row: label_Age(row),axis=1)

test_df['SexNum'] = test_df.apply(lambda row: label_Sex(row),axis=1)

test_df['EmbarkedNum'] = test_df.apply(lambda row: label_Port(row),axis=1)

test_df['FareNum'] = test_df.apply(lambda row: label_Fare(row),axis=1)



dfAuxSur = test_df[['Pclass', "SexNum", 'SibSp', 'Parch', 'FareNum', 'EmbarkedNum', 'AgeNum']]



matrizPropiedadesSur = dfAuxSur.values



datosTest = np.array(matrizPropiedadesSur)



vectorResTest = model.predict(datosTest) #vectorResTest es el vector de resultados
#Aquí creo el archivo que subí: 

submissionFrame = pd.DataFrame({'PassengerId':np.array(test_df['PassengerId'].values), 'Survived':vectorResTest})

submissionFrame.head(1000)

submissionFrame.set_index('PassengerId')

submissionFrame.to_csv('submission.csv', index=False)

submissionFrame.head(1000)
#Así decidí cuál usar: 

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="linear", C=0.025),

    SVC(gamma=2, C=1),

    GaussianProcessClassifier(1.0 * RBF(1.0)),

    DecisionTreeClassifier(max_depth=5),

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

    MLPClassifier(alpha=1),

    AdaBoostClassifier(),

    GaussianNB(),

    QuadraticDiscriminantAnalysis(), LogisticRegression(), LinearSVC(), Perceptron(), SGDClassifier(), DecisionTreeClassifier() ]



names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",

         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",

         "Naive Bayes", "QDA", "Logistic Regression", "Linear SVC", "Perceptron", "SGDClassifier", "DecisionTree"]



for name, clf in zip(names, classifiers):

        clf.fit(X_train, y_train)

        score = clf.score(X_test, y_test)

        pp(score)

        
