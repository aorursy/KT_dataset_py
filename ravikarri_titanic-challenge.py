#Just Started ML. Please rate and Advice. 





# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from sklearn.cluster import KMeans

from sklearn.cross_validation import train_test_split

from sklearn.neighbors import KNeighborsClassifier

#from sklearn.ensemble import RandomForestRegressor

from sklearn import preprocessing

from sklearn.preprocessing import Imputer

from sklearn.metrics import accuracy_score

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
def get_titles():



    global combined

    

    # we extract the title from each name

    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    

    # a map of more aggregated titles

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

    

    # we map each title

    combined['Title'] = combined.Title.map(Title_Dictionary)
tit = pd.read_csv("../input/train.csv")

#print(tit)

print(tit.columns)

#print(tit.Survived)

le = preprocessing.LabelEncoder()

le.fit(tit['Sex'])

tit['Sex'] = le.transform(tit['Sex'].get_values())

tit['Fam'] = tit['SibSp'] + tit['Parch']

tit['Alone'] = np.where(tit['Fam'] > 0, 0, 1)

#print(tit.head())

tit['Embarked'] = tit['Embarked'].fillna('C')

#print (list(tit['Embarked']))

C_O_I = ['Pclass', 'Fare','Sex','Fam','Embarked','Alone']

X = tit[C_O_I]

le2 = preprocessing.LabelEncoder()

le2.fit(X['Embarked'])

X['Embarked'] = le2.fit_transform(X['Embarked'].astype(str))

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

imp = imp.fit(X)

X = imp.transform(X)

#print(X.head())

Y = tit['Survived']

#print(list(Y))
X_train, X_test, y_train, y_test = train_test_split(

 X, Y, test_size = 0.3, random_state = 100)

y_train = y_train.ravel()

y_test = y_test.ravel()
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

imp = imp.fit(X)

X = imp.transform(X)

#for K in range(25):

 #K_value = K+1

neigh = KNeighborsClassifier(n_neighbors = 1, weights='uniform', algorithm='auto')

 #neigh.fit(X_train, y_train) 

 #y_pred = neigh.predict(X_test)

 #print("Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:",K_value)

neigh.fit(X, Y) 

    
for K in range(600):

 K_value = K+1

 neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')

 neigh.fit(X_train, y_train) 

 y_pred = neigh.predict(X_test)

 print ("Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:",K_value)
#Please don't Run this 

T_model = KMeans(n_clusters=2, random_state=0)

# Fit model

T_model.fit(X, Y)

test = pd.read_csv("../input/test.csv")

#print(test.head())

test['Embarked'] = test['Embarked'].fillna('C')

test['Fam'] = test['SibSp'] + test['Parch']

test['Alone'] = np.where(test['Fam'] > 0, 0, 1)

X = test[C_O_I]

#print(set(list(X['Sex'])))

#X['Sex'] = le.transform(X['Sex'].get_values())

X['Embarked'] = le.fit_transform(X['Embarked'].astype(str))

X['Sex'] = le.fit_transform(X['Sex'].astype(str))

print(X.head())

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

imp = imp.fit(X)

X = imp.transform(X)

L = neigh.predict(X)

L = [ 0 if i < 0.5 else 1 for i in L ]

L1 = list(test['PassengerId'])

for i in range(len(L)):

    print(str(L1[i])+","+str(L[i]))