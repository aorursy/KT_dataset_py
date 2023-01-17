import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

import sklearn

from sklearn import metrics



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.model_selection import train_test_split

train= pd.read_csv('/kaggle/input/titanic/train.csv')

test= pd.read_csv('/kaggle/input/titanic/test.csv')

combine = [train, test]

#train和test数据的读取
for dataset in combine:    

    dataset['Age'] = dataset.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

    

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)



    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

drop_column = ['PassengerId','Cabin', 'Ticket']

train= train.drop(drop_column, axis=1)

test= test.drop(['Cabin', 'Ticket'], axis=1)

combine = [train,test]

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train['Title'], train['Sex'])

for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train = train.drop(['Name'], axis=1)

test = test.drop(['Name'], axis=1)

combine = [train, test]
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
for dataset in combine:    

    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1 

    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0
train['FareBin'] = pd.qcut(train['Fare'], 4)

test['FareBin'] = pd.qcut(test['Fare'], 4)

train['AgeBin'] = pd.cut(train['Age'].astype(int), 5)

test['AgeBin'] = pd.cut(test['Age'].astype(int), 5)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train = train.drop(['FareBin'], axis=1)

test = test.drop(['FareBin'], axis=1)

combine = [train, test]

for dataset in combine:

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] >16) & (dataset['Age'] <= 32), 'Age']= 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age']= 2

    dataset.loc[ dataset['Age'] > 54, 'Age'] = 3

    dataset['Age'] = dataset['Age'].astype(int)

train = train.drop(['AgeBin'], axis=1)

test = test.drop(['AgeBin'], axis=1)

combine = [train, test]
X = train.drop("Survived", axis=1)

Y = train["Survived"]

Test_X  = test.drop("PassengerId", axis=1).copy()

Train_X , Valid_X , Train_Y , Valid_Y = train_test_split( X, Y , train_size = .7 )
modelKNN = KNeighborsClassifier(n_neighbors = 3)

modelKNN.fit(Train_X, Train_Y)

Y_predValid = modelKNN.predict(Valid_X)

AccModel = round(modelKNN.score(Train_X, Train_Y) * 100, 2)

AccValidKNN = round((sum(Valid_Y == Y_predValid)/Valid_Y.count())*100,2)



maxAcc = AccValidKNN

model = modelKNN

print(AccValidKNN)
A = [0.05, 0.1, 0.3, 0.7, 1, 1.5, 2, 3]

B = [0.05, 0.1, 0.3, 0.7, 1, 1.5, 2, 3]

AccM = np.zeros((len(A), len(B)))

acc = np.zeros((len(A), len(B)))

Cideal = 0

Gamideal = 0

for i in range(len(A)):

    for j in range(len(B)):

               modelSVC = SVC(C = A[i],gamma = B[j])

               modelSVC.fit(Train_X, Train_Y)

               Y_predValid = modelSVC.predict(Valid_X)

               AccM[i,j] = round(modelSVC.score(Train_X, Train_Y) * 100, 2)

               acc[i,j] = round((sum(Valid_Y == Y_predValid)/Valid_Y.count())*100,2)

               if (acc[i,j] >= np.amax(acc)): 

                    Cideal = A[i]

                    Gamideal = B[j]

        

modelSVC = SVC(C = Cideal, gamma = Gamideal)

modelSVC.fit(Train_X, Train_Y)

Y_predValid = modelSVC.predict(Valid_X)

AccModel = round(modelSVC.score(Train_X, Train_Y) * 100, 2)

AccValidSVC = round((sum(Valid_Y == Y_predValid)/Valid_Y.count())*100,2)

print(AccValidSVC)



if (AccValidSVC > maxAcc): 

    maxAcc = AccValidSVC

    model = modelSVC
modelDT = DecisionTreeClassifier()

modelDT.fit(Train_X, Train_Y)

Y_predValid = modelDT.predict(Valid_X)

AccModel = round(modelDT.score(Train_X, Train_Y) * 100, 2)

AccValidDT = round((sum(Valid_Y == Y_predValid)/Valid_Y.count())*100,2)

print(AccValidDT)

if (AccValidDT > maxAcc): 

    maxAcc = AccValidDT

    model = modelDT


modelRF = RandomForestClassifier()

modelRF.fit(Train_X, Train_Y)

Y_predValid = modelRF.predict(Valid_X)

AccModel = round(modelRF.score(Train_X, Train_Y) * 100, 2)

AccValidRF = round((sum(Valid_Y == Y_predValid)/Valid_Y.count())*100,2)

print(AccValidRF)

if (AccValidRF > maxAcc): 

    maxAcc = AccValidRF

    model = modelRF
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN',  'Random Forest',  'Decision Tree'],

    'Score': [AccValidSVC, AccValidKNN,AccValidRF, AccValidDT]})

models.sort_values(by='Score', ascending=False)
sorted_model=models.sort_values(by='Score', ascending=False)

plt.figure(figsize=(20,10))

fig = plt.bar(sorted_model['Model'], sorted_model['Score'],color='aqua')

plt.grid()

plt.show()
Test_Y = model.predict( Test_X )

passenger_id = test.PassengerId

Final = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': Test_Y } )

Final.shape

Final.head(n=20)

Final.to_csv( '预测结果.csv' , index = False )