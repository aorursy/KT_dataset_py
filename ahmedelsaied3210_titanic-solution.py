# data analysis and wrangling

import pandas as pd

import numpy as np



# visualization

from pandas.tools.plotting import scatter_matrix

from matplotlib import pyplot

%matplotlib inline



# machine learning

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
full_test = pd.read_csv('../input/test.csv')
train.shape
train.head()
test.shape
test.head()
train.columns
print(train.describe())
train.corr()
train.isnull().sum(axis = 0)
test.isnull().sum(axis = 0)
def mean_age_finder(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age
train['Age'] = train[['Age', 'Pclass']].apply(mean_age_finder, axis = 1)
most_frequent_embarked = train['Embarked'].value_counts().index[0]      #Output: 'S'

train['Embarked'].fillna(most_frequent_embarked, inplace = True)
train.isnull().sum()
del train['Cabin']
train.isnull().sum()
# Finding the mean age of each passenger class:

import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize = (12, 8))

sns.boxplot(x = 'Pclass', y = 'Age', data = test)
def mean_age_finder_test(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 42

        elif Pclass == 2:

            return 27

        else:

            return 24

    else:

        return Age
test['Age'] = test[['Age', 'Pclass']].apply(mean_age_finder_test, axis = 1)
test.isnull().sum()
#Replacing the missing value of 'Fare' column with mean value:



test['Fare'].fillna(test['Fare'].mean(), inplace = True)
del test['Cabin']
test.isnull().sum()
sex = pd.get_dummies(train['Sex'], drop_first = True)

embarked = pd.get_dummies(train['Embarked'], drop_first = True)
train.drop(['Name', 'Sex', 'Ticket', 'Embarked'], axis = 1, inplace = True)

train = pd.concat([train, sex, embarked], axis = 1)
train.head(2)
sex = pd.get_dummies(test['Sex'], drop_first = True)

embarked = pd.get_dummies(test['Embarked'], drop_first = True)
test.drop(['Name', 'Sex', 'Ticket', 'Embarked'], axis = 1, inplace = True)

test = pd.concat([test, sex, embarked], axis = 1)
test.head()
del train['PassengerId']

del test['PassengerId']
del train['Fare']

del test['Fare']
X=train.copy()

del X['Survived']

y=train['Survived']

X_test=test.copy()

print(X.shape)

print(y.shape)

print(X_test.shape)
X.head()
print(train.groupby('Survived').size())
X.plot(kind='box',subplots=True,layout=(2,4),sharey=False)

pyplot.show()
X.hist()

pyplot.show()
scatter_matrix(X)

pyplot.show()
print(X.shape)

print(y.shape)

print(train.shape)
X_train ,X_val,y_train,y_val=train_test_split(X,y,test_size=0.20,random_state=7)
models=[]

models.append(('LR',LogisticRegression()))

models.append(('LDA',LinearDiscriminantAnalysis()))

models.append(('KNN',KNeighborsClassifier()))

models.append(('CART',DecisionTreeClassifier()))

models.append(('NB',GaussianNB()))

models.append(('SVM',SVC()))



results=[]

names=[]

for name,model in models:

    kfold=KFold(n_splits=10,random_state=42)

    cv_result=cross_val_score(model,X_train,y_train,cv=kfold,scoring='accuracy')

    results.append(cv_result)

    names.append(name)

    msg="%s: %f (%f)" % (name,cv_result.mean(),cv_result.std())

    print(msg)
svm=SVC()

svm.fit(X_train,y_train)

linear_svc = LogisticRegression()

linear_svc.fit(X_train, y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train,y_train) * 100, 2)

acc_linear_svc
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)

dtree_predictions = dtree.predict(X_test)



dtree_accuracy = round(dtree.score(X_train, y_train) * 100, 2)

print('Decision Tree Model Accuracy: ', dtree_accuracy)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 100)

rfc.fit(X_train, y_train)

rfc_predictions = rfc.predict(X_test)



rfc_accuracy = round(rfc.score(X_train, y_train) * 100, 2)

print('Random Forest Model Accuracy: ', rfc_accuracy)
submission = pd.DataFrame({

        "PassengerId": full_test["PassengerId"],

        "Survived": rfc_predictions

    })

submission.to_csv('submission.csv',index = False)