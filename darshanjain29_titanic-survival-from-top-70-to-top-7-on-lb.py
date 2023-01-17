# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import cross_val_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train_df  = pd.read_csv("/kaggle/input/titanic/train.csv")
print ("*"*10, "Dataset information", "*"*10)
print (train_df.info())
print ("*"*10, "First 5 test rows", "*"*10)
print (train_df.head(5))
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
print ("*"*10, "Dataset information", "*"*10)
print (test_df.info())
print ("*"*10, "Last 5 test rows", "*"*10)
print (test_df.tail(5))
train_df.Cabin.value_counts()
train_df.Cabin.value_counts()

train_df["Age"].isnull().sum()
#Age null fix
data = [train_df, test_df]

for dataset in data:
    mean = train_df["Age"].mean()
    #td = test_df["Age"].std()
    #is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    #rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    #age_slice = dataset["Age"].copy()
    #age_slice[np.isnan(age_slice)] = rand_age
    #dataset["Age"] = age_slice
    
    dataset['Age'] = dataset['Age'].fillna(mean)
    dataset["Age"] = train_df["Age"].astype(float)
train_df["Age"].isnull().sum()
'''
test_df  = pd.read_csv("/kaggle/input/titanic/test.csv")
data = [train_df, test_df]

for dataset in data:
    dataset['relatives']= dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int) 

train_df['not_alone'].value_counts()

#Drop passenger ID as it is not required
train_df = train_df.drop(['PassengerId'], axis=1)
'''


#Data processing
#1. In the cabin variable, create new column and add there only first letters of the column
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [train_df, test_df]

for dataset in data:
    dataset['Deck'] = dataset['Cabin'].fillna("U")
    dataset['Deck'] = dataset['Cabin'].astype(str).str[0] 
    dataset['Deck'] = dataset['Deck'].str.capitalize()
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int) 

train_df['Deck'].value_counts()

#Dropping Cabin feature
train_df = train_df.drop(['Cabin'], axis=1)
test_df.drop(['Cabin'], axis=1, inplace = True)
print (train_df['Deck'].value_counts())
train_df.info()
'''
train_df['Age'].fillna(train_df['Age'].mode(), inplace=True)
test_df['Age'].fillna(test_df['Age'].mode(), inplace=True)
#train_df["Age"].isnull().sum()
print (train_df["Age"].isnull().sum())
'''
train_df.Sex.value_counts()
#print (train_df['Embarked'].value_counts())

common_value = 'S'
data = [train_df, test_df]
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)

train_df.info()
train_df.head(10)
data = [train_df, test_df] 
embarkedMap = {"S": 0, "C": 1, "Q": 2}

genderMap = {"male": 0, "female": 1}

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    #dataset['Fare'] = dataset['Fare'].astype(int) 
    dataset['Embarked'] = dataset['Embarked'].map(embarkedMap)
    dataset['Sex'] = dataset['Sex'].map(genderMap)
    #print (dataset['Embarked'])
    
train_df['Sex'].describe()
train_df['Embarked'].describe()
train_df.info()
train_df.head(10)

#Title extraction
data = [train_df, test_df]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

#Braund, Mr. Owen Harris
for dataset in data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.')
    #print(dataset['Title'])
    
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Mlle', 'Ms', 'Mme'], 'Rare')

    dataset['Title'] = dataset['Title'].map(titles)
    # filling NaN with 0, to get safe
    dataset['Title'] = dataset['Title'].fillna(0)
    
    
print (train_df['Title'].value_counts())


train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
train_df['Title'].value_counts()


train_df.info()
train_df['Ticket'].value_counts()
train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1) 

data = [train_df, test_df]
for dataset in data:
    #dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6
    
# let's see how it's distributed 
train_df['Age'].value_counts()

data = [train_df, test_df]

#train_df['category_fare'] = pd.qcut(train_df['Fare'], 4)

#train_df['category_fare'].value_counts()

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(float)

train_df['Fare'].value_counts()

'''
data = [train_df, test_df]
for dataset in data:
    dataset['Age_Class']= dataset['Age']* dataset['Pclass']
    '''
'''
for dataset in data:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
# Let's take a last look at the training set, before we start training the models.
train_df.head(10)
'''
train_df.info()
'''
train_df  = train_df.drop("Ticket", axis=1)
test_df  = test_df.drop("Ticket", axis=1)
'''
train_df = train_df.drop(['PassengerId'], axis=1)
X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']

test_df.head(10)
X_test = test_df.drop("PassengerId", axis=1).copy()
X_test.head(10)
X_train.info()
'''
train_df  = train_df.drop("not_alone", axis=1)
test_df  = test_df.drop("not_alone", axis=1)

train_df  = train_df.drop("Parch", axis=1)
test_df  = test_df.drop("Parch", axis=1)
'''



from sklearn.linear_model import LogisticRegression

clf = LogisticRegression() 
clf.fit(X_train, Y_train)

Y_pred  = clf.predict(X_test)

scores = cross_val_score(clf, X_train, Y_train, cv = 10, scoring = "accuracy")

#clf.score(X_train, Y_train)
#acc_logistic_reg = round(clf.score(X_train, Y_train)*100, 2)

print ("Scores: ",scores)
print ("Mean: ", scores.mean())
print ("Standard Deviation: ", scores.std())

# 2. SVM
from sklearn import svm
clf_svm = svm.SVC()
clf_svm.fit(X_train, Y_train)

Y_pred_svm  = clf_svm.predict(X_test)

scores_svm = cross_val_score(clf_svm, X_train, Y_train, cv = 10, scoring = "accuracy")
print ("Scores: ",scores_svm.mean())
# 2. Decision tree

from sklearn import tree
clf_dt = tree.DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=5, min_samples_split=25,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')
clf_dt.fit(X_train, Y_train)

Y_pred_svm  = clf_dt.predict(X_test)

scores_dt = cross_val_score(clf_dt, X_train, Y_train, cv = 10, scoring = "accuracy")
print ("Scores: ",scores_dt.mean())
# 2. Random forest

from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(max_depth=2, random_state=0)

clf_rf.fit(X_train, Y_train)

Y_pred_rf  = clf_rf.predict(X_test)

scores_rf = cross_val_score(clf_rf, X_train, Y_train, cv = 10, scoring = "accuracy")
print ("Scores: ",scores_rf.mean())
# 2. Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier

clf_gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

clf_gbc.fit(X_train, Y_train)

Y_pred_rf  = clf_gbc.predict(X_test)

scores_rf = cross_val_score(clf_gbc, X_train, Y_train, cv = 10, scoring = "accuracy")
print ("Scores: ",scores_rf.mean())


# 2. Bagging Classifier

from sklearn.ensemble import BaggingClassifier

clf_bagging = BaggingClassifier()

clf_bagging.fit(X_train, Y_train)

Y_pred_rf  = clf_bagging.predict(X_test)

scores_rf = cross_val_score(clf_bagging, X_train, Y_train, cv = 10, scoring = "accuracy")
print ("Scores: ",scores_rf.mean())
# 2. Naive Bayes - Gaussian

from sklearn.naive_bayes import GaussianNB

clf_gnb = GaussianNB()

clf_gnb.fit(X_train, Y_train)

Y_pred_rf  = clf_gnb.predict(X_test)

scores_rf = cross_val_score(clf_gnb, X_train, Y_train, cv = 10, scoring = "accuracy")
print ("Scores: ",scores_rf.mean())


import xgboost as xgb
from xgboost import XGBClassifier

clf_xgb = XGBClassifier().fit(X_train, Y_train)

Y_pred  = clf_xgb.predict(X_test)

clf_xgb.score(X_train, Y_train)

scores_rf = cross_val_score(clf_xgb, X_train, Y_train, cv = 10, scoring = "accuracy")
print ("Scores: ",scores_rf.mean())

from sklearn.model_selection import cross_val_score

random_forest = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=5, min_samples_split=16,
                       min_weight_fraction_leaf=0.0, n_estimators=700,
                       n_jobs=-1, oob_score=True, random_state=1, verbose=0,
                       warm_start=False)
scores = cross_val_score(random_forest, X_train, Y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

param_grid = { 
    "criterion" : ["gini", "entropy"], 
    "min_samples_leaf" : [1, 5, 10, 25, 50, 70, 90, 120], 
    "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35, 45, 55, 65]}

dt = DecisionTreeClassifier()

#clf = GridSearchCV(dt, param_grid=param_grid)
clf = RandomizedSearchCV(dt, param_grid)
 
clf.fit(X_train, Y_train)
 
print(clf.best_estimator_)

'''from sklearn.model_selection import cross_val_score

clf_dt = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=10, min_samples_split=35,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')
scores = cross_val_score(clf_dt, X_train, Y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())'''
'''#Hyper parameter tuining - Decision trees
param_grid = { 
    "criterion" : ["gini", "entropy"], 
    "min_samples_leaf" : [1, 5, 10, 25, 50, 70], 
    "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35], 
    "n_estimators": [100, 400, 700, 1000, 1500]}

rf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=1, n_jobs=-1)

#clf = GridSearchCV(rf, param_grid=param_grid, n_jobs=-1)
clf = RandomizedSearchCV(rf, param_grid, n_jobs=-1)
 
clf.fit(X_train, Y_train)
print(clf.best_estimator_)'''
#print(clf.best_estimator_)
'''params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

#param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10, 25, 50, 70], "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35], "n_estimators": [100, 400, 700, 1000, 1500]}

xgb = XGBClassifier(learning_rate=0.02, n_estimators=600,
                    silent=True, nthread=1)

#rf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=1, n_jobs=-1)

clf = GridSearchCV(xgb, param_grid=params)
#clf = RandomizedSearchCV(xgb, param_grid, n_jobs=-1)
 
clf.fit(X_train, Y_train)

print(clf.best_estimator_)'''

'''# Logistic regression before HPT
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train)*100,2)

print (acc_log)
'''
'''from sklearn.model_selection import cross_val_score
logreg = LogisticRegression()
scores = cross_val_score(logreg, X_train, Y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())'''
'''
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': Y_pred})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
'''