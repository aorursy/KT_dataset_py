# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import sklearn 

import warnings

warnings.filterwarnings("ignore")



from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import MinMaxScaler,LabelEncoder,OneHotEncoder,StandardScaler

from sklearn.model_selection import train_test_split,cross_val_score, KFold

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB, MultinomialNB,BernoulliNB

from sklearn.svm import LinearSVC, SVC

from sklearn import metrics

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier



%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#creating database

dados=pd.read_csv("/kaggle/input/xAPI-Edu-Data/xAPI-Edu-Data.csv")

dados
#Data visualization

dados.info()
#bar chart for the gender variable

dados["gender"].value_counts().plot.bar(title='Number of students by gender')
#bar chart for the Topic variable

dados["Topic"].value_counts().plot.bar(title='Most popular topic')
#bar chart for the Place of Birth variable

dados["PlaceofBirth"].value_counts().plot.bar(title='Most present nationalities')
#bar chart for the StageID variable

dados["StageID"].value_counts().plot.bar(title='School level')
#histogram for Discussion variable

dados["Discussion"].value_counts().plot.hist(bins=15, title='Number of discussion')
#creating predictors and target variable 

x=dados.loc[:,['raisedhands','VisITedResources','AnnouncementsView','Discussion','StudentAbsenceDays']].values

y=dados.loc[:,'Class'].values
x=pd.DataFrame(x)

y=pd.DataFrame(y)
#Label encoding of features

x=x.apply(LabelEncoder().fit_transform)

y=y.apply(LabelEncoder().fit_transform)
#Scaling

scaler = StandardScaler()

x = scaler.fit_transform(x)
#Creating a test and training dataset

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30,random_state = 0)
#Applying classification algorithms



# Logistic Regression

Regressor = LogisticRegression()

Regressor.fit(X_train,y_train)

previsoes = Regressor.predict(X_test)



matriz =  confusion_matrix(y_test,previsoes)

accuracy = accuracy_score(y_test,previsoes)

precision = precision_score(y_test,previsoes,average='macro')

recall = recall_score(y_test,previsoes,average='macro')

f1 = f1_score(y_test,previsoes,average='macro')



print(matriz)

print(accuracy)

print(precision)

print(recall)

print(f1)
# Naive Bayes 

classificador = GaussianNB()

classificador.fit(X_train,y_train)

previsoes = classificador.predict(X_test)

 

matriz =  confusion_matrix(y_test,previsoes)

accuracy = accuracy_score(y_test,previsoes)

precision = precision_score(y_test,previsoes,average='macro')

recall = recall_score(y_test,previsoes,average='macro')

f1 = f1_score(y_test,previsoes,average='macro')



print(matriz)

print(accuracy)

print(precision)

print(recall)

print(f1)
# Linear SVM model 

classificador = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, C=10)

classificador.fit(X_train,y_train)

previsoes = classificador.predict(X_test)

 

matriz =  confusion_matrix(y_test,previsoes)

accuracy = accuracy_score(y_test,previsoes)

precision = precision_score(y_test,previsoes,average='macro')

recall = recall_score(y_test,previsoes,average='macro')

f1 = f1_score(y_test,previsoes,average='macro')



print(matriz)

print(accuracy)

print(precision)

print(recall)

print(f1)
# KNN

classificador = KNeighborsClassifier(n_neighbors = 10)

classificador.fit(X_train,y_train)

previsoes = classificador.predict(X_test)

 

matriz =  confusion_matrix(y_test,previsoes)

accuracy = accuracy_score(y_test,previsoes)

precision = precision_score(y_test,previsoes,average='macro')

recall = recall_score(y_test,previsoes,average='macro')

f1 = f1_score(y_test,previsoes,average='macro')



print(matriz)

print(accuracy)

print(precision)

print(recall)

print(f1)
# Random Forest

classificador = RandomForestClassifier()

classificador.fit(X_train,y_train)

previsoes = classificador.predict(X_test)

 

matriz =  confusion_matrix(y_test,previsoes)

accuracy = accuracy_score(y_test,previsoes)

precision = precision_score(y_test,previsoes,average='macro')

recall = recall_score(y_test,previsoes,average='macro')

f1 = f1_score(y_test,previsoes,average='macro')



print(matriz)

print(accuracy)

print(precision)

print(recall)

print(f1)
