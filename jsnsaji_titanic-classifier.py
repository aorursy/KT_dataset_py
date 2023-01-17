#classifiers

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

#preprocessing

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

import numpy as np

import pandas as pd



#evaluation

from sklearn.model_selection import cross_val_score

from scipy.stats.stats import pearsonr
#load data

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
train.head(5)
train.corr()
#dealing with missing values in 'Age'

train.Age.fillna(train.Age.mean(), inplace=True)



#dealing with missing values in 'embarked'

#print(len(X.Embarked))

train.Embarked.fillna('S', inplace=True)
#label encoding for gender

le = preprocessing.LabelEncoder()

le.fit(pd.unique(train.Sex))

#le.classes_ for printing distinct classes

sex_t = le.transform(train.Sex)

test_sex_t = le.transform(test.Sex)



#label encoding for embarked

le = preprocessing.LabelEncoder()

le.fit(pd.unique(train.Embarked))

embarked_t = le.transform(train.Embarked)

test_embarked_t = le.transform(test.Embarked)
Y = train.iloc[:,1]

train = train.iloc[:,[2,4,5,6,7,9,11]]

test = test.iloc[:,[1,3,4,5,6,8,10]]





train.iloc[:,1] = sex_t

test.iloc[:,1] = test_sex_t



train.iloc[:,6] = embarked_t

test.iloc[:,6] = test_embarked_t





print(train.head(3))

print(test.head(3))
#normalize/scale data

train = train.fillna(method='ffill')

test = test.fillna(method='ffill')

scaler = StandardScaler()

scaler.fit(train.values)

X_train = scaler.transform(train)

X_test = scaler.transform(test)
# pearson's coefficient

pclassco = pearsonr(Y, X_train[:,0])

sexco = pearsonr(Y, X_train[:,1])

ageco = pearsonr(Y, X_train[:,2])

sibspco = pearsonr(Y, X_train[:,3])

parchco = pearsonr(Y, X_train[:,4])

embarkedco = pearsonr(Y, X_train[:,5])

family = pearsonr(Y,X_train[:,3] + X_train[:,4])

print(pclassco)

print(sexco)

print(ageco)

print(sibspco)

print(parchco)

print(embarkedco)

print(family)
#svm with grid search

svm = SVC()

parameters = {'kernel':('linear', 'rbf'), 'C':(1,0.25,0.5,0.75),'gamma': (1,2,3,'auto'),'decision_function_shape':('ovo','ovr'),'shrinking':(True,False)}

clf = GridSearchCV(svm, parameters)

clf.fit(X_train,Y)

print("accuracy:"+str(np.average(cross_val_score(clf, X_train, Y, scoring='accuracy'))))

print("f1:"+str(np.average(cross_val_score(clf, X_train, Y, scoring='f1'))))
