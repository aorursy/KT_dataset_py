%matplotlib inline



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
train = pd.read_csv("../input/train.csv",header=0)

test = pd.read_csv("../input/test.csv",header=0)
train.head()
test.head()
trainY = train[['Survived']]

colsTrain = train.columns.tolist() #Make a list of all of the columns in the df

colsTrain.pop(colsTrain.index('Survived')) #Remove 'Survived' from list

train = train[colsTrain] #Create new dataframe with columns in the order you want



#train.head()
titanic = pd.concat([train,test])
titanic.isnull().any()
titanic.info()
for i in [1,2,3]:

    femaleAgeMean = titanic.loc[(titanic['Pclass']==i) & (titanic['Sex']=='female')].Age.mean()

    titanic.ix[(titanic['Age'].isnull()) & (titanic['Pclass']==i) & (titanic['Sex']=='female'), 'Age'] = femaleAgeMean

    maleAgeMean = titanic.loc[(titanic['Pclass']==i) & (titanic['Sex']=='male')].Age.mean()

    titanic.ix[(titanic['Age'].isnull()) & (titanic['Pclass']==i) & (titanic['Sex']=='male'), 'Age'] = maleAgeMean

    fareMean = titanic.loc[titanic['Pclass']==i].Fare.mean()

    titanic.ix[(titanic['Fare'].isnull()) & (titanic['Pclass']==i), 'Fare'] = fareMean
titanic.Sex = titanic.Sex.map({'male': 1, 'female': 2})
titanic.Embarked.unique()

mapEmbarked = {k: v for v, k in enumerate(titanic.Embarked.dropna().unique(),2)}

mapEmbarked = {**{np.nan:1}, **mapEmbarked}

titanic.Embarked = titanic.Embarked.map(mapEmbarked)
titanic["DuplicatedTicket"] = titanic.Ticket.duplicated(keep=False).map({False: 0, True: 1})
titanic.loc[titanic['Ticket']=="113803"]
# Number of duplicated tickets -> Number of 'families' in the ship?

len(titanic.Ticket.unique())
mapping = {k: v for v, k in enumerate(titanic.Ticket.unique(),1)}

titanic['Ticket'] = titanic.Ticket.map(mapping)
titanic['CabinDeck'] = titanic.Cabin.str[0]

titanic.CabinDeck.unique()
mapCabinDeck = {k: v for v, k in enumerate(titanic.CabinDeck.dropna().unique(),2)}

mapCabinDeck = {**{np.nan:1}, **mapCabinDeck}

titanic['CabinDeck'] = titanic.CabinDeck.map(mapCabinDeck)
mapCabin = {k: v for v, k in enumerate(titanic.Cabin.dropna().unique(),2)}

mapCabin = {**{np.nan: 0},**mapCabin}

titanic['Cabin'] = titanic.Cabin.map(mapCabin)
titanic.sample(5)
titanic['Title'] = (titanic.Name.str.split('.').str.get(0)).str.split(', ').str.get(1)+'.'

titanic.Title.unique()
mapTitle = {k: v for v, k in enumerate(titanic.Title.unique(),1)}

titanic['Title'] = titanic.Title.map(mapTitle)
titanic['Surname'] = (titanic.Name.str.split(',').str.get(0))

mapSurname = {k: v for v, k in enumerate(titanic.Surname.unique(),1)}

titanic['Surname'] = titanic.Surname.map(mapSurname)
titanic = titanic.drop('Name', 1)

titanic = titanic.drop('PassengerId', 1)

titanic.head()
X_train, X_cv, y_train, y_cv = train_test_split(titanic[0:891][:], trainY, test_size=0.4, random_state=42)

X_cv, X_test, y_cv, y_test = train_test_split(X_cv, y_cv, test_size=0.5, random_state=42)



scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_cv = scaler.transform(X_cv)

X_test = scaler.transform(X_test)



NN_arch = [(100,), (10,10), (10,10,10), (10,10,10,10), (20,20), (20,20,20), (20,20,20,20), (30,30), (30,30,30), (30,30,30,30)]

regularization = [0.000001, 0.000003, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]

accuracy = np.zeros((len(NN_arch),len(regularization)))

for i in range(len(NN_arch)):

    for j in range(len(regularization)):

        clf = MLPClassifier(hidden_layer_sizes=NN_arch[i], activation='logistic', alpha=regularization[j],max_iter=20000)



        clf.fit(X_train, np.ravel(y_train))



        accuracy[i,j] = clf.score(X_cv,y_cv)

    

#print(accuracy)
# Choose the most accurate combination of NN_arch/regularization

id = np.unravel_index(accuracy.argmax(),accuracy.shape)

# Define the classifier

clf = MLPClassifier(hidden_layer_sizes=NN_arch[id[0]], activation='logistic', alpha=regularization[id[1]],max_iter=20000)

# Fit now the whole training set

X_train = titanic[0:891][:]

y_train = trainY

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = titanic[891:][:]

X_test = scaler.transform(X_test)



clf.fit(X_train, np.ravel(y_train))



y_test = clf.predict(X_test)
surv = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_test})



surv.to_csv('predictionTitanicNeuralNetwork.csv',index=False)
X_train, X_cv, y_train, y_cv = train_test_split(titanic[0:891][:], trainY, test_size=0.4, random_state=42)

X_cv, X_test, y_cv, y_test = train_test_split(X_cv, y_cv, test_size=0.5, random_state=42)



scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_cv = scaler.transform(X_cv)

X_test = scaler.transform(X_test)



gamma = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]

coef0 = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]

accuracy = np.zeros((len(gamma),len(coef0)))

for i in range(len(gamma)):

    for j in range(len(coef0)):



        clf = svm.SVC(C=1.0, kernel='sigmoid', gamma=gamma[i], coef0=coef0[j], shrinking=True, probability=False,

              tol=0.0001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None,

              random_state=None)

        

        clf.fit(X_train, np.ravel(y_train))



        accuracy[i,j] = clf.score(X_cv,y_cv)





#print(accuracy)
# Choose the most accurate combination of NN_arch/regularization

id = np.unravel_index(accuracy.argmax(),accuracy.shape)

# Define the classifier

clf = svm.SVC(C=1.0, kernel='sigmoid', gamma=gamma[id[0]], coef0=coef0[id[1]], shrinking=True, probability=False,

              tol=0.0001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None,

              random_state=None)

# Fit now the whole training set

X_train = titanic[0:891][:]

y_train = trainY

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = titanic[891:][:]

X_test = scaler.transform(X_test)



clf.fit(X_train, np.ravel(y_train))



y_test = clf.predict(X_test)
surv = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_test})



surv.to_csv('predictionTitanicSVM.csv',index=False)