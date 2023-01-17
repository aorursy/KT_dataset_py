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
#reading data

df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
#checking how data looks like

df.head()
#Plotting librarys

import matplotlib.pyplot as plt

import seaborn as sns
plt.hist(df['target'])

plt.title('Target distribution')

plt.show()
df['target'].value_counts()
#age X target

ax= sns.boxplot(x = df.target, y = df.age)

plt.title('Age X Target')

plt.show()
#Correlation between sex, age and the target

sns.catplot(x="sex", y="age",kind="swarm", hue="target" , data=df)

plt.title('Sex X Age X Target')

plt.show()
df.isnull().any()
df.describe()
df.info()
y = df.target

X = df.drop(columns = "target")

print(y.shape)

print(X.shape)
#Stantarding

from sklearn import preprocessing



scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
#Split in to train and test

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#Getting validation dataset

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
#Models

from sklearn.linear_model import LogisticRegression #logistic regression

from sklearn.svm import SVC  #support vector Machine

from sklearn.tree import DecisionTreeClassifier #Decision Tree

from sklearn.ensemble import RandomForestClassifier #Random Forest

from sklearn.neighbors import KNeighborsClassifier #KNN

from sklearn.naive_bayes import GaussianNB #Naive bayes



#Cross validation

from sklearn.model_selection import GridSearchCV



#Metrics

from sklearn import metrics #accuracy measure

from sklearn.metrics import confusion_matrix, classification_report #for confusion matrix
#LogisticRegression

modelLR = LogisticRegression()
# Define the grid of hyperparameters to search

hyperparameter_grid = {

    'penalty':['l2', 'none'], 

    'fit_intercept':[False, True], 

    'solver':['newton-cg', 'lbfgs', 'sag', 'saga'], 

    'max_iter':[50, 75, 100, 125], 

}
clfLR = GridSearchCV(modelLR, hyperparameter_grid, cv=5, n_jobs=-1)
clfLR.fit(X_train, y_train)
validationLR = clfLR.predict(X_val)

print('The validation score of the Logistic Regression is',metrics.accuracy_score(validationLR,y_val))

confusion_matrix(y_val,validationLR)
predictionLR = clfLR.predict(X_test)

print('The testing score of the Logistic Regression is',metrics.accuracy_score(predictionLR,y_test))

print(classification_report(y_test, predictionLR))

print('\n')

print(confusion_matrix(y_test,predictionLR))
#SVM

modelSVM = SVC()

hyperparameter_grid = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

clfSVM = GridSearchCV(modelSVM, hyperparameter_grid, cv=10)





clfSVM.fit(X_train, y_train)
validationSVM = clfSVM.predict(X_test)

print('The test score of the Logistic Regression is',metrics.accuracy_score(validationLR,y_val))

print(classification_report(y_test, validationSVM))

print('\n')

print(confusion_matrix(y_test,validationSVM))
#DecisionTreeClassifier

modelDT = DecisionTreeClassifier()

modelDT.fit(X_train, y_train)
validationDT = modelDT.predict(X_test)

print('The test score of the Logistic Regression is',metrics.accuracy_score(validationLR,y_val))

print(classification_report(y_test, validationDT))

print('\n')

print(confusion_matrix(y_test,validationDT))
#RandomForestClassifier

modelRF = RandomForestClassifier()

modelRF.fit(X_train, y_train)
predRF = modelRF.predict(X_test)

#print('The test score of the Logistic Regression is',metrics.accuracy_score(validationLR,y_val))

print(classification_report(y_test, predRF))

print('\n')

print(confusion_matrix(y_test,predRF))
#KNN

modelKNN = KNeighborsClassifier()

modelKNN.fit(X_train, y_train)
valKNN = modelKNN.predict(X_val)

print(classification_report(y_val, valKNN))

print('\n')

print(confusion_matrix(y_val,valKNN))
predKNN = modelKNN.predict(X_test)

print(classification_report(y_test, predKNN))

print('\n')

print(confusion_matrix(y_test,predKNN))
#Naive Bayes

modelNB = GaussianNB()

modelNB.fit(X_train, y_train)
valNB = modelNB.predict(X_val)

print(classification_report(y_val, valNB))

print('\n')

print(confusion_matrix(y_val,valNB))
predNB = modelNB.predict(X_test)

print(classification_report(y_test, predNB))

print('\n')

print(confusion_matrix(y_test,predNB))