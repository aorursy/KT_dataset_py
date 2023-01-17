# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from sklearn import preprocessing,cross_validation,svm

from sklearn.linear_model import LinearRegression,LogisticRegression

from sklearn import tree

from sklearn.ensemble import GradientBoostingRegressor,RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier #KNN

from sklearn.naive_bayes import GaussianNB #Naive bayes
df = pd.read_csv('../input/train.csv')

print(df.head())
def training_data():

    X = df[['Sex','SibSp','Parch','Fare','Age','Embarked']]

    X=X.fillna(method='ffill')



    X.loc[X['Sex']=='female','Sex']=1

    X.loc[X['Sex']=='male','Sex']=0

    X.loc[X['Embarked']=='S','Embarked']=-1

    X.loc[X['Embarked']=='C','Embarked']=0

    X.loc[X['Embarked']=='Q','Embarked']=1



    print(X.head())

    X_train = preprocessing.scale(X)

    y_train = df['Survived']

    return X_train,y_train

X,y = training_data()
def testing_data():

    df = pd.read_csv('../input/test.csv')

    X = df[['Pclass','Sex','SibSp','Parch','Fare','Age','Embarked']]

    X=X.fillna(-99999)



    X.loc[X['Sex']=='female','Sex']=1

    X.loc[X['Sex']=='male','Sex']=0

    X.loc[X['Embarked']=='S','Embarked']=-1

    X.loc[X['Embarked']=='C','Embarked']=0

    X.loc[X['Embarked']=='Q','Embarked']=1

    print(X.head())

    X_test = preprocessing.scale(X)

    return X_test

test = testing_data()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
#Linear Support Vector Machine(linear-SVM)

clf = svm.SVC()

clf = clf.fit(X_train, y_train)

clf.fit(X_train,y_train)

clf.score(X_test,y_test)
#Radial Support Vector Machines(rbf-SVM)

clf = svm.SVC(kernel='sigmoid',C=10,gamma=0.05)

clf = clf.fit(X_train, y_train)

clf.fit(X_train,y_train)

clf.score(X_test,y_test)
#Logistic Regression



clf = LogisticRegression()

clf = clf.fit(X_train, y_train)

clf.fit(X_train,y_train)

clf.score(X_test,y_test)
#decision tree



clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)

clf.fit(X_train,y_train)

clf.score(X_test,y_test)
#K-Nearest Neighbours(KNN)

clf = KNeighborsClassifier()

clf = clf.fit(X_train, y_train)

clf.fit(X_train,y_train)

clf.score(X_test,y_test)
#Gaussian Naive Bayes

clf = GaussianNB()

clf = clf.fit(X_train, y_train)

clf.fit(X_train,y_train)

clf.score(X_test,y_test)
def get_output():  

    clf = svm.SVC(kernel='rbf',C=1,gamma=0.05)

    X,y = training_data()

    clf = clf.fit(X,y)

    clf.fit(X_train,y_train)

    out = pd.DataFrame({'PassengerId':[i for i in range(892,1310)]})

    out['Survived'] = clf.predict(testing_data())

    out.set_index('PassengerId',inplace=True)

    out.to_csv('out.csv')

    print(out)