# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def submission_generation(dataframe, name):

    """

    Esta función genera un csv a partir de un dataframe de pandas. 

    Con FileLink se genera un enlace desde el que poder descargar el fichero csv

    

    dataframe: DataFrame de pandas

    name: nombre del fichero csv

    """

    import os

    from IPython.display import FileLink

    os.chdir(r'/kaggle/working')

    dataframe.to_csv(name, index = False)

    return  FileLink(name)
Train = pd.read_csv("/kaggle/input/rms2-titanic/train.csv")

Test = pd.read_csv("/kaggle/input/rms2-titanic/test.csv")
# Combinamos abos data set y le añadimos una nueva columna donde queará el titulo

train_test_data = [Train,Test]



for dataset in train_test_data:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')
for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', \

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

Train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#Ahora comvertimos a las mujeres en 1 y a los hombres en 0

for dataset in train_test_data:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
#A los pasajeros que no tengan definida la columna 'Embarked' se les asignará el puerto S

for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')
#Ahora cambiamos los datos de la columna 'Embarked' S , C y Q por 0 , 1 y 2

for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
#Rellenamos las valores NaN de la columna edad con valores aleatorios entre media, mediana

for dataset in train_test_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)
#Ponemos valores numericos a intervalos de edad

for dataset in train_test_data:

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
#Creamos una nueva columna en el Dataset 'FamilySize' 

for dataset in train_test_data:

    dataset['FamilySize'] = dataset['Sibsp'] +  dataset['Parch'] + 1
#Creamos una nueva columna llamada 'IsAlone'

for dataset in train_test_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
Train.head()
t1=pd.get_dummies(Train["Title"], prefix='Titulo: ')

t2=pd.get_dummies(Test["Title"], prefix='T2')



Train=pd.concat([Train,t1], axis=1)

Test=pd.concat([Test,t2], axis=1)
t3=pd.get_dummies(Train["Age"], prefix='Rando de edad')

t4=pd.get_dummies(Test["Age"], prefix='Rando de edad')



Train=pd.concat([Train,t3], axis=1)

Test=pd.concat([Test,t4], axis=1)
Train.head()
t5=pd.get_dummies(Train["FamilySize"], prefix='Tamaño de familia : ')

t6=pd.get_dummies(Test["FamilySize"], prefix='Tamaño de familia : ')



Train=pd.concat([Train,t5], axis=1)

Test=pd.concat([Test,t6], axis=1)
t7=pd.get_dummies(Train["Pclass"], prefix='Clase nº : ')

t8=pd.get_dummies(Test["Pclass"], prefix='Clase nº : ')



Train=pd.concat([Train,t7], axis=1)

Test=pd.concat([Test,t8], axis=1)
#Quitamos las columnas que no nos interesan, las que no van a dar información relevante

features_drop = ['Name', 'Sibsp', 'Parch', 'Ticket', 'Cabin', 'Fare','PassengerId','Title','IsAlone','FamilySize','Pclass']

Train = Train.drop(features_drop, axis=1)

TestDataFrame=Test

Test = Test.drop(features_drop, axis=1)
Test.head()
#Preparamos los datos

X_train=Train.drop(['Survived'], axis=1)

Y_train=Train['Survived']

X_test=Test
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X_train, Y_train)

y_pred_log_reg = clf.predict(X_test)

acc_log_reg = round( clf.score(X_train, Y_train) * 100, 2)

print (str(acc_log_reg) + ' percent')
RegresiónLogisticaDataFrame=pd.DataFrame({"PassengerId":TestDataFrame.PassengerId,

                                          "Survived":y_pred_log_reg})



submission_generation(RegresiónLogisticaDataFrame, "PrediccionRegresionLogistica.csv")
from sklearn.svm import SVC, LinearSVC

clf = SVC()

clf.fit(X_train, Y_train)

y_pred_svc = clf.predict(X_test)

acc_svc = round(clf.score(X_train, Y_train) * 100, 2)

print (acc_svc)

SVMDataFrame=pd.DataFrame({"PassengerId":TestDataFrame.PassengerId,

                           "Survived":y_pred_svc})



submission_generation(SVMDataFrame, "PrediccionSVM.csv")
from sklearn.svm import SVC, LinearSVC

clf = LinearSVC()

clf.fit(X_train, Y_train)

y_pred_linear_svc = clf.predict(X_test)

acc_linear_svc = round(clf.score(X_train, Y_train) * 100, 2)

print (acc_linear_svc)
SVM_LinearDataFrame=pd.DataFrame({"PassengerId":TestDataFrame.PassengerId,

                                  "Survived":y_pred_linear_svc})



submission_generation(SVM_LinearDataFrame, "PrediccionSVM_Linear.csv")
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors = 3)

clf.fit(X_train, Y_train)

y_pred_knn = clf.predict(X_test)

acc_knn = round(clf.score(X_train, Y_train) * 100, 2)

print (acc_knn)
K_NearestNeighbrosDataFrame=pd.DataFrame({"PassengerId":TestDataFrame.PassengerId,

                                          "Survived":y_pred_knn})



submission_generation(K_NearestNeighbrosDataFrame, "PrediccionKNN.csv")
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

clf.fit(X_train, Y_train)

y_pred_decision_tree = clf.predict(X_test)

acc_decision_tree = round(clf.score(X_train, Y_train) * 100, 2)

print (acc_decision_tree)
DecisionTreeDataFrame=pd.DataFrame({"PassengerId":TestDataFrame.PassengerId,

                                    "Survived":y_pred_decision_tree})



submission_generation(DecisionTreeDataFrame, "PrediccionDecisionTree.csv")
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train, Y_train)

y_pred_random_forest = clf.predict(X_test)

acc_random_forest = round(clf.score(X_train, Y_train) * 100, 2)

print (acc_random_forest)
RandomForestDataFrame=pd.DataFrame({"PassengerId":TestDataFrame.PassengerId,

                                    "Survived":y_pred_random_forest})



submission_generation(RandomForestDataFrame, "PrediccionRandomForest.csv")
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

clf.fit(X_train, Y_train)

y_pred_gnb = clf.predict(X_test)

acc_gnb = round(clf.score(X_train, Y_train) * 100, 2)

print (acc_gnb)
GaussianNaiveDataFrame=pd.DataFrame({"PassengerId":TestDataFrame.PassengerId,

                                     "Survived":y_pred_gnb})



submission_generation(GaussianNaiveDataFrame, "PrediccionGNB.csv")
from sklearn.linear_model import Perceptron

clf = Perceptron(max_iter=5, tol=None)

clf.fit(X_train, Y_train)

y_pred_perceptron = clf.predict(X_test)

acc_perceptron = round(clf.score(X_train, Y_train) * 100, 2)

print (acc_perceptron)
PerceptronDataFrame=pd.DataFrame({"PassengerId":TestDataFrame.PassengerId,

                                  "Survived":y_pred_perceptron})



submission_generation(PerceptronDataFrame, "PrediccionPerceptron.csv")
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(max_iter=5, tol=None)

clf.fit(X_train, Y_train)

y_pred_sgd = clf.predict(X_test)

acc_sgd = round(clf.score(X_train, Y_train) * 100, 2)

print (acc_sgd)
SGDDataFrame=pd.DataFrame({"PassengerId":TestDataFrame.PassengerId,

                           "Survived":y_pred_sgd})



submission_generation(SGDDataFrame, "PrediccionSGD.csv")
models = pd.DataFrame({

    'Model': ['Logistic Regression', 'Support Vector Machines', 'Linear SVC', 

              'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes', 

              'Perceptron', 'Stochastic Gradient Decent'],

    

    'Score': [acc_log_reg, acc_svc, acc_linear_svc, 

              acc_knn,  acc_decision_tree, acc_random_forest, acc_gnb, 

              acc_perceptron, acc_sgd]

    })



models.sort_values(by='Score', ascending=False)