# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')
data.head()
data.describe()
data.Survived.plot('hist')
data.dropna().Survived.plot('hist')
data.Fare.plot('hist')
data.Fare = data.Fare.fillna(data.Fare.median())
data.dropna().Survived.plot('hist')
sel_data = data[['Age','Fare', 'Pclass', 'Survived', 'Parch', 'SibSp', 'Sex']]
sel_data.dropna().Survived.plot('hist')
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
Sex_lb = LabelBinarizer()
sel_data['Label_Sex'] = Sex_lb.fit_transform(sel_data.Sex)
sel_data

sel_data.Age.plot('hist')
sel_data.Age = sel_data.Age.fillna(np.round(sel_data.Age.mean()))
Age_lb = LabelEncoder()
Age_bins = [0, 10, 20, 40, 60, 100]
sel_data['Age_bin'] = Age_lb.fit_transform(pd.cut(sel_data.Age, Age_bins).apply(lambda X: X.right))
Fare_lb = LabelEncoder()
Fare_temp, Fare_bins = pd.cut(sel_data.Fare, 5, retbins = True)
sel_data['Fare_bin'] = Fare_lb.fit_transform(Fare_temp.apply(lambda X: X.right))
X = sel_data.dropna().drop(['Sex','Survived', 'Fare','Age'], axis =1).values
Y =  sel_data.dropna().Survived.values
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = 'lbfgs')
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .1)
lr.fit(X_train, Y_train)
from sklearn import metrics
def print_metrics(clf, X_test, Y_test ):
    y_pred = clf.predict(X_test)
    
    print('accuracy:  ', metrics.accuracy_score(Y_test, y_pred))
    print('precision: ', metrics.precision_score(Y_test, y_pred))
    print('recall:    ', metrics.recall_score(Y_test, y_pred))
    print('f1-score:  ', metrics.f1_score(Y_test, y_pred))
print_metrics(lr, X_test, Y_test)
from sklearn.tree import ExtraTreeClassifier
ada = AdaBoostClassifier(ExtraTreeClassifier())
ada.fit(X_train, Y_train)
print_metrics(ada, X_test, Y_test)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, weights = 'distance')
knn.fit(X_train, Y_train)
print_metrics(knn, X_test, Y_test)
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier([50,40,30,20,15,14,13,12,11])
mlp.fit(X_train, Y_train)

print_metrics(mlp, X_test, Y_test)
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
mnb = MultinomialNB()
mnb.fit(X_train, Y_train)
print_metrics(mnb, X_test, Y_test)
bnb = BernoulliNB()
bnb.fit(X_train, Y_train)
print_metrics(bnb, X_test, Y_test)
from sklearn.svm import NuSVC, SVC, OneClassSVM
nsvc = OneClassSVM()
nsvc.fit(X_train[Y_train == 1], Y_train[Y_train == 1])
print_metrics(nsvc, X_test[Y_test==1], Y_test[Y_test == 1])
svc = SVC(probability=True)
svc.fit(X_train, Y_train)
print_metrics(svc, X_test, Y_test)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
print_metrics(rfc, X_test, Y_test)
vc = VotingClassifier([('bnb', bnb), ('lr', lr),('rfc', rfc)], voting = 'soft')
vc.fit(X_train, Y_train)
print_metrics(vc, X_test, Y_test)
test_data = pd.read_csv('../input/test.csv')
test_data
test_data.Age.describe()
test_data['Label_Sex'] = Sex_lb.transform(test_data.Sex)
test_data['Age_bin'] = Age_lb.transform(pd.cut(test_data.Age.fillna(np.round(test_data.Age.mean())), Age_bins).apply(lambda X: X.right))
test_data['Fare_bin'] = Fare_lb.transform(pd.cut(test_data.Fare.fillna(test_data.Fare.median()), Fare_bins).apply(lambda X: X.right))

test_data = test_data[list(sel_data.drop(['Survived'], axis=1).columns)+['PassengerId']]
test_data.drop(['Sex', 'Fare','Age', 'PassengerId'], axis =1).head()
sel_data.head()
XTest = test_data.drop(['Sex', 'Fare','Age', 'PassengerId'], axis =1).values
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier([100,100,50,50,25,25,15,15,10,10,5,5], activation='tanh')
mlp.fit(X_train, Y_train)
print_metrics(mlp, X_test, Y_test)
test_data['Survived'] = ada.predict(XTest)
sub = test_data[['PassengerId', 'Survived']]
sub = sub.set_index('PassengerId')
sub.head()
sub.to_csv('titanic_sub.csv')





























