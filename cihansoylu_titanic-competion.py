# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')
data_train.head()
data_train.describe()
# Which features has missing values?
data_train.apply(pd.isna).apply(np.any)
#Clean the data

#Categorize the age feature after filling the missing values
def categorize_ages(df):
    df.Age = df.Age.fillna(-.5)
    bins = (-1, 0, 5, 12, 18, 30, 55, 81 )
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']
    df.Age = pd.cut(df.Age, bins, labels=group_names)  
    return df

#Simplify cabin number to a single letter after filling the missing values
def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')  #fill NaN with N
    df.Cabin = df.Cabin.apply(lambda x: x[0]) #Replace with the first letter
    return df

#Drop irrelevant features
def drop_features(df):   
    return df.drop(['Ticket', 'Name', 'Embarked', 'Cabin'], axis=1)

#Categorize the fares after filling the missing values
def categorize_fares(df):
    df.Fare = df.Fare.fillna(-1)
    bins = (-2,0, 8, 15, 31, 52, 75, 93, 115, 135, 1000)
    group_names = ["unknown", "quartile_0", "quartile_1", "quartile_2", "quartile_3", "quartile_4", "quartile_5", 
                   "quartile_6", "quartile_7", "quartile_8"]
    df.Fare = pd.cut(df.Fare, bins, labels=group_names)
    return df

#Categorize class
def categorize_pclass(df):
    bins = (.5, 1.5, 2.5, 3.5)
    group_names = ["first_class", "second_class", "third_class"]
    df.Pclass = pd.cut(df.Pclass, bins, labels=group_names)
    return df


def transform_features(df):
    df = categorize_ages(df)
    #df = simplify_cabins(df)
    df = categorize_fares(df)
    df = drop_features(df)
    df = categorize_pclass(df)
    return df
data_test = transform_features(data_test)
data_train = transform_features(data_train)
data_train.head()
data_train = pd.get_dummies(data_train)
data_test = pd.get_dummies(data_test)
from sklearn.model_selection import train_test_split

X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
y_all = data_train['Survived']

num_test = 0.30
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23, stratify = y_all)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
#clf = GaussianNB()
#clf = KNeighborsClassifier(n_neighbors=5)
#clf = LogisticRegression()
#clf = SVC()
#clf = RandomForestClassifier()
clf = tree.DecisionTreeClassifier()

#param_grid = {'C': [0.0001, 0.001, 0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.1, 1, 10, 100]}
#param_grid = {'C' : [.001, .01, .1, 1, 10, 100], 'tol' : [.00001, .0001, .001, .01, .1]}
#param_grid = {'n_estimators' : [10, 15, 20, 30], 'max_depth' : [2, 5, 10, 15, 20]}
param_grid = {'max_depth' : [3, 5, 10, 20, 30], 'min_samples_split': [2, 3, 4, 5 ]}
#param_grid = {}

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 42)
grid_search = GridSearchCV(clf, param_grid, cv = skf)
grid_search.fit(X_train, y_train)
print('Grid Search score:', grid_search.score(X_test, y_test))
clf = grid_search.best_estimator_
print(clf)


#clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print('Accuracy score:', clf.score(X_test, y_test))
print('Training Accuracy', clf.score(X_train, y_train))
print('Precision score', precision_score(y_test, pred))
print('Recall score', recall_score(y_test, pred))

#precision, recall, thresholds = precision_recall_curve(labels_test, clf.predict_proba(features_test)[:,1])
#plt.plot(precision, recall, label="precision recall curve")
#plt.xlabel("Precision")
#plt.ylabel("Recall")


my_predictions = clf.predict(data_test.drop('PassengerId', axis = 1))
submission = pd.DataFrame({'PassengerId': data_test.PassengerId, 'Survived': my_predictions})
submission.to_csv('submission.csv', index=False)
