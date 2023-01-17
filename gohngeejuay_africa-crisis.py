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
import warnings

warnings.simplefilter("ignore")
crises = pd.read_csv("../input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv")
crises.head()
crises.shape
crises.describe()
#Checking for null values

crises[crises.isna().any(axis = 1)]
#Checking for null values

crises.isna().sum()
#Checking correlation between factors to systemic_crisis

crises.drop(['case'],axis = 1).corr()

crises.iloc[:,[3,5,6,7,8,9,10,11,12]].apply(lambda x : x.corr(crises['systemic_crisis']))   
#Separating the features and target

x = crises.drop(['case','cc3','country','systemic_crisis'],axis = 1)    #drop irrelevant columns

y = crises['systemic_crisis']#crises.loc[crises['systemic_crisis']]
#Explanatory variables: All variables excluding case,cc3,country,systemic_crisis columns

x
#Dependent variable: systemic_crisis

y
#Convert banking_crisis column into encoding using one hot encoding

#Two different ways of creating dummy variables

#x = pd.get_dummies(x)  #Create two column for banking_crisis(one for crisis, another for no_crisis)

x = pd.get_dummies(x,drop_first = True) #Only one column for banking_crisis(0 = crisis, 1 = no_crisis)

x
#Standardization

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_transformed = scaler.fit_transform(x)    #result as a numpy array

x = pd.DataFrame(x_transformed, index = x.index, columns = x.columns )    #keep as a dataframe

x
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)

logReg = LogisticRegression()

logReg.fit(X_train,y_train)
from sklearn.metrics import confusion_matrix

y_pred = logReg.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

print(cm)

print("Accuracy = " + str(metrics.accuracy_score(y_test,y_pred)))    #sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
print(logReg.coef_)

print(x.columns)

for i in range(len(logReg.coef_[0])):    

    print("Coeeficient of " + x.columns[i] + " : " + str(logReg.coef_[0][i]))
import math

#Convert log-odds into odds

odds = []

for i in range(len(logReg.coef_[0])):

    odds.append(math.exp(logReg.coef_[0][i]))

#print(odds)



#Convert odds into probability

probs = []

for i in range(len(odds)):

    probs.append(odds[i]/(1+odds[i]))

#print(probs)

for i in range(len(probs)):    

    print("Coeeficient of " + x.columns[i] + " : " + str(probs[i]))
#GaussianNB

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

print(cm)

print("Accuracy of GaussianNB = " + str(metrics.accuracy_score(y_test,y_pred)))    #sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
#BernoulliNB

from sklearn.naive_bayes import BernoulliNB

bnb = BernoulliNB()

bnb.fit(X_train,y_train)

y_pred = bnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

print(cm)

print("Accuracy BernoulliNB = " + str(metrics.accuracy_score(y_test,y_pred)))    #sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
#MultinomialNB

# from sklearn.naive_bayes import MultinomialNB

# mnb = MultinomialNB()

# mnb.fit(X_train,y_train)

# y_pred = mnb.predict(X_test)

# cm = confusion_matrix(y_test,y_pred)

# print(cm)

# print("Accuracy MultinomialNB = " + str(metrics.accuracy_score(y_test,y_pred)))    #sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)



#ValueError: Input X must be non-negative
#ComplementNB

# from sklearn.naive_bayes import ComplementNB

# cnb = ComplementNB()

# cnb.fit(X_train,y_train)

# y_pred = cnb.predict(X_test)

# cm = confusion_matrix(y_test,y_pred)

# print(cm)

# print("Accuracy ComplementNB = " + str(metrics.accuracy_score(y_test,y_pred)))    #sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)



#ValueError: Input X must be non-negative
# CategoricalNB

# from sklearn.naive_bayes import CategoricalNB

# cnb = CategoricalNB()

# cnb.fit(X_train,y_train)

# y_pred = cnb.predict(X_test)

# cm = confusion_matrix(y_test,y_pred)

# print(cm)

# print("Accuracy CategoricalNB = " + str(metrics.accuracy_score(y_test,y_pred)))    #sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)



#ImportError: cannot import name 'CategoricalNB'
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

print(cm)

print("Accuracy knn 5 neighbor : " + str(metrics.accuracy_score(y_test,y_pred)))    #sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)

accuracy_result = []

for i in range(5,30,5):

    knn = KNeighborsClassifier(n_neighbors = i)

    knn.fit(X_train,y_train)

    y_pred = knn.predict(X_test)

    accuracy_result.append(metrics.accuracy_score(y_test,y_pred))

print(accuracy_result)

import matplotlib.pyplot as plt

plt.plot([5,10,15,20,25],accuracy_result)
from sklearn import svm

#linear kernel

linsvm = svm.SVC(kernel = "linear")

linsvm.fit(X_train,y_train)

y_pred = linsvm.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

print(cm)

print("Accuracy linear kernel svm : " + str(metrics.accuracy_score(y_test,y_pred)))
#Different kernels: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

#rbf kernel

rbfsvm = svm.SVC(kernel = "rbf")

rbfsvm.fit(X_train,y_train)

y_pred = rbfsvm.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

print(cm)

print("Accuracy rbf kernel svm : " + str(metrics.accuracy_score(y_test,y_pred)))
#Different kernels: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

#poly kernel

polysvm = svm.SVC(kernel = "poly")

polysvm.fit(X_train,y_train)

y_pred = polysvm.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

print(cm)

print("Accuracy poly kernel svm : " + str(metrics.accuracy_score(y_test,y_pred)))
#sigmoid kernel

sigsvm = svm.SVC(kernel = "sigmoid")

sigsvm.fit(X_train,y_train)

y_pred = sigsvm.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

print(cm)

print("Accuracy sigmoid kernel svm : " + str(metrics.accuracy_score(y_test,y_pred)))
# #precomputed kernel

# presvm = svm.SVC(kernel = "precomputed")

# presvm.fit(X_train,y_train)

# y_pred = presvm.predict(X_test)

# cm = confusion_matrix(y_test,y_pred)

# print(cm)

# print("Accuracy precomputed kernel svm : " + str(metrics.accuracy_score(y_test,y_pred)))



#ValueError: X.shape[0] should be equal to X.shape[1]
#Regularization parameter: C.Must be positive float.

cValues = [0.01,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.5,3.0,4.0,5.0,6.0]

results = []

for i in range(len(cValues)):

    #poly kernel

    polysvm = svm.SVC(kernel = "poly", C = cValues[i])

    polysvm.fit(X_train,y_train)

    y_pred = polysvm.predict(X_test)

    cm = confusion_matrix(y_test,y_pred)

    print("Accuracy poly kernel svm with C value = " + str(cValues[i]) + " : "+ str(metrics.accuracy_score(y_test,y_pred)))

    results.append(metrics.accuracy_score(y_test,y_pred))

plt.plot(cValues,results)
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=100)

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

print(cm)

print("Accuracy Random Forest : " + str(metrics.accuracy_score(y_test,y_pred)))
for i in range(len(clf.feature_importances_)):    

    print("Coeeficient of " + x.columns[i] + " : " + str(clf.feature_importances_[i]))
