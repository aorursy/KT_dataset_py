import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import os





%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
train.describe()
dataset = pd.concat((train, test))
dataset = dataset.fillna(np.nan)

dataset.isnull().sum()
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].mean())
dataset["Embarked"] = dataset["Embarked"].fillna("S")
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]

dataset["Title"] = pd.Series(dataset_title)

dataset["Title"].value_counts()
dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

dataset["Title"] = dataset["Title"].astype(int)
dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1

dataset['Single'] = dataset['FamilySize'].map(lambda s: 1 if s == 1 else 0)

dataset['SmallF'] = dataset['FamilySize'].map(lambda s: 1 if  s == 2  else 0)

dataset['MedF'] = dataset['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

dataset['LargeF'] = dataset['FamilySize'].map(lambda s: 1 if s >= 5 else 0)
dataset['Cabin'].describe()
dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])
Ticket = []

for i in list(dataset.Ticket):

    if not i.isdigit() :

        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0])

    else:

        Ticket.append("X")

        

dataset["Ticket"] = Ticket

dataset["Ticket"].head()
dataset = pd.get_dummies(dataset, columns = ["Sex", "Title"])

dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")

dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix="T")

dataset["Pclass"] = dataset["Pclass"].astype("category")

dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")

dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")
dataset.drop(labels = ["Name"], axis = 1, inplace = True)

dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)
dataset.shape
X_train = dataset[:train.shape[0]]

X_test = dataset[train.shape[0]:]

y = train['Survived']
X_train = X_train.drop(labels='Survived', axis=1)

X_test = X_test.drop(labels='Survived', axis=1)
from sklearn.preprocessing import StandardScaler
headers_train = X_train.columns

headers_test = X_test.columns
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

from sklearn.model_selection import GridSearchCV
cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25, random_state = 0 )

accuracies = cross_val_score(LogisticRegression(solver='liblinear'), X_train, y, cv  = cv)
from sklearn.svm import SVC



C = [0.1, 1]

gammas = [0.01, 0.1]

kernels = ['rbf', 'poly']

param_grid = {'C': C, 'gamma' : gammas, 'kernel' : kernels}



cv = StratifiedShuffleSplit(n_splits=5, test_size=.25, random_state=8)



grid = GridSearchCV(SVC(probability=True), param_grid, cv=cv)

grid.fit(X_train,y)
svm_grid= grid.best_estimator_

svm_score = round(svm_grid.score(X_train,y), 4)

print('Accuracy for SVM: ', svm_score)
from sklearn.neighbors import KNeighborsClassifier
k_range = range(1,31)

weights_options=['uniform','distance']

param = {'n_neighbors':k_range, 'weights':weights_options}

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

grid = GridSearchCV(KNeighborsClassifier(), param,cv=cv,verbose = False, n_jobs=-1)



grid.fit(X_train,y)
knn_grid= grid.best_estimator_

knn_score = round(knn_grid.score(X_train,y), 4)

knn_score

print('Accuracy for KNN: ', knn_score)
C_vals = [0.2,0.3,0.4,0.5,1,5,10]



penalties = ['l1','l2']



cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25)





param = {'penalty': penalties, 'C': C_vals}



logreg = LogisticRegression(solver='liblinear')

 

grid = GridSearchCV(estimator=LogisticRegression(), 

                           param_grid = param,

                           scoring = 'accuracy',

                            n_jobs =-1,

                           cv = cv

                          )



grid.fit(X_train, y)
logreg_grid = grid.best_estimator_

logreg_score = round(logreg_grid.score(X_train,y), 4)

print('Accuracy for Logistic Regression: ', logreg_score)
predict = knn_grid.predict(X_test)

submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predict})

submission.to_csv('submission_knn.csv', index=False)