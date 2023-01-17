
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import math
import xgboost

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/gender_submission.csv')

train.head()
#Does the passanger have a cabin?
train['cabin_binary'] = train["Cabin"].apply(lambda i: 0 if str(i) == "nan" else 1)

#Family Size
train['family_size'] = 1 + train['SibSp'] + train['Parch']
train['solo'] = train["family_size"].apply(lambda i: 1 if i == 1 else 0)

#Fix Nulls
train['Embarked'] = train['Embarked'].fillna('S')
train['Age'] = train['Age'].fillna(int(np.mean(train['Age'])))

#A few age specific Binaries
train['Child'] = train["Age"].apply(lambda i: 1 if i <= 17 and i > 6 else 0)
train['toddler'] = train["Age"].apply(lambda i: 1 if i <= 6 else 0)
train['Elderly'] = train["Age"].apply(lambda i: 1 if i >= 60 else 0)

# Fancy fancy
train['fancy'] = train['Fare'].apply(lambda i: 1 if i >= 100 else 0)

# standard
train['standard_fare'] = train['Fare'].apply(lambda i: 1 if i <= 10.0 else 0)

#No requirement to standardize in DT models, but might as well
fare_scaler = StandardScaler()
fare_scaler.fit(train['Fare'].values.reshape(-1, 1))
train['fare_std'] = fare_scaler.transform(train['Fare'].values.reshape(-1, 1))

#get status of passanger
train['title'] = 'default'

for i in train.values:
    name = i[3] #First checks for rare titles (Thanks Anisotropic's wonderful Kernel for inspiration//help here!)
    for e in ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:
        if e in name:
            train.loc[train['Name'] == name, 'title'] = 'rare'
    if 'Miss' in name or  'Mlle' in name or 'Ms' in name or 'Mme' in name or 'Mrs' in name:
        train.loc[train['Name'] == name, 'title'] = 'Ms'
    if 'Mr.' in name or 'Master' in name:
        train.loc[train['Name'] == name, 'title'] = 'Mr'


train.head(10)
#Does the passanger have a cabin?
test['cabin_binary'] = test["Cabin"].apply(lambda i: 0 if str(i) == "nan" else 1)

#Family Size
test['family_size'] = 1 + test['SibSp'] + test['Parch']
test['solo'] = test["family_size"].apply(lambda i: 1 if i == 1 else 0)

#Fix Nulls
test['Embarked'] = test['Embarked'].fillna('S')
test['Age'] = test['Age'].fillna(int(np.mean(test['Age'])))

#A few age specific Binaries
test['Child'] = test["Age"].apply(lambda i: 1 if i <= 17 and i > 6 else 0)
test['toddler'] = test["Age"].apply(lambda i: 1 if i <= 6 else 0)
test['Elderly'] = test["Age"].apply(lambda i: 1 if i >= 60 else 0)

# Fancy fancy
test['fancy'] = test['Fare'].apply(lambda i: 1 if i >= 100 else 0)
test['standard_fare'] = test['Fare'].apply(lambda i: 1 if i <= 10.0 else 0)

#standardize
test['fare_std'] = fare_scaler.transform(test['Fare'].values.reshape(-1, 1))

#get status of passanger
test['title'] = 'default'

for i in test.values:
    name = i[2] #First checks for rare titles (Thanks Anisotropic's wonderful Kernel for inspiration//help here!)
    for e in ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:
        if e in name:
            test.loc[test['Name'] == name, 'title'] = 'rare'
    if 'Miss' in name or  'Mlle' in name or 'Ms' in name or 'Mme' in name or 'Mrs' in name:
        test.loc[test['Name'] == name, 'title'] = 'Ms'
    if 'Mr.' in name or 'Master' in name:
        test.loc[test['Name'] == name, 'title'] = 'Mr'


test.head(10)
train = pd.get_dummies(train, columns=["Sex", "Embarked", "title"])
test = pd.get_dummies(test, columns=["Sex", "Embarked", "title"])

train = train.drop(['Name','PassengerId', 'Ticket', 'Cabin', 'Fare', 'SibSp'], axis = 1)
test = test.drop(['Name','PassengerId', 'Ticket', 'Cabin', 'Fare', 'SibSp'], axis = 1)

train.head()
#COR MATRIX OF numerical vars
plt.figure(figsize=(14,12))
plt.title('Corelation Matrix', size=12)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=plt.cm.RdBu, linecolor='white', annot=True)
#Family size histo
sns.distplot(train['family_size'])
#boxplot of family size and survival
sns.boxplot("Survived", y="family_size", data = train)
#Fare to Age relationship?
sns.lmplot('fare_std', 'Age', data = train, 
           fit_reg=False,scatter_kws={"marker": "D", "s": 20}) 

#boxplot of age and survival
sns.boxplot("Survived", "Age", data = train)
#bar chart of age and survival
sns.lmplot("Survived", "fare_std", data = train, fit_reg=False)
X = train.drop(['Survived'], axis = 1)
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Random Forest Setup
ranfor = RandomForestClassifier()
parameters = {'n_estimators':[10,50,100], 'random_state': [42, 138], \
              'max_features': ['auto', 'log2', 'sqrt']}
ranfor_clf = GridSearchCV(ranfor, parameters)
ranfor_clf.fit(X_train, y_train)


'''CROSS VALIDATE'''
cv_results = cross_validate(ranfor_clf, X_train, y_train)
cv_results['test_score']  

y_pred = ranfor_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
##Decision Tree Go
dt = DecisionTreeClassifier()
parameters = {'random_state': [42, 138],'max_features': ['auto', 'log2', 'sqrt']}
dt_clf = GridSearchCV(dt, parameters)
dt_clf.fit(X_train, y_train)

y_pred = dt_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
ada = AdaBoostClassifier(base_estimator = DecisionTreeClassifier())
parameters = {'n_estimators':[10,50,100], 'random_state': [42, 138], 'learning_rate': [0.1, 0.5, 0.8, 1.0]}
ada_clf = GridSearchCV(ada, parameters)
ada_clf.fit(X_train, y_train)

cv_results = cross_validate(ada_clf, X_train, y_train)
cv_results['test_score']  

y_pred = ada_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
gradBoost = GradientBoostingClassifier()
parameters = {'n_estimators':[10,50,100], 'random_state': [42, 138], 'learning_rate': [0.1, 0.5, 0.8, 1.0], \
             'loss' : ['deviance', 'exponential']}
gb_clf = GridSearchCV(gradBoost, parameters)
gb_clf.fit(X_train, y_train)

cv_results = cross_validate(gb_clf, X_train, y_train)
cv_results['test_score']  

y_pred = gb_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
xg = xgboost.XGBClassifier(max_depth = 3, n_estimators = 600, learning_rate = 0.05)
xg.fit(X_train, y_train)

cv_results = cross_validate(xg, X_train, y_train)
cv_results['test_score']  

y_pred = xg.predict(X_test)
print(accuracy_score(y_test, y_pred))
'''Confusion Matrix'''
y_pred = xg.predict(X_test)
# TN, FP, FN, TP
confusion_matrix(y_test, y_pred)
predictions = xg.predict(test)

sub['Survived'] = predictions
sub.to_csv("first_submission_xgb.csv", index=False)