# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import optimize
from matplotlib import pyplot
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/titanic/train.csv")
train['fam_size']=train['SibSp']+train['Parch']
train['Ticket_Len'] = train['Ticket'].apply(lambda x: len(x))           #obtaining ticket no. length and cabin no. length from the given ticket numbers
train['Ticket_Lett'] = train['Ticket'].apply(lambda x: str(x)[0])
train['Cabin_Letter'] = train['Cabin'].apply(lambda x: str(x)[0])
train.head()



test = pd.read_csv("/kaggle/input/titanic/test.csv")
test['fam_size']=test['SibSp']+test['Parch']

test['Ticket_Len'] = test['Ticket'].apply(lambda x: len(x))
test['Ticket_Lett'] = test['Ticket'].apply(lambda x: str(x)[0])
test['Cabin_Letter'] = test['Cabin'].apply(lambda x: str(x)[0])







train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
train['Ticket_Len'].fillna(train['Ticket_Len'].mode()[0], inplace=True)
train['Ticket_Lett'].fillna(train['Ticket_Lett'].mode()[0],inplace=True)
train['Cabin_Letter'].fillna(train['Cabin_Letter'].mode()[0],inplace=True)

train['Age'].fillna(train['Age'].mean(),inplace=True)
train['Sex'].fillna(train['Sex'].mode()[0],inplace=True)

train['fam_size'].fillna(train['fam_size'].mode()[0],inplace=True)
train['Fare'].fillna(train['Fare'].mean(),inplace=True)
train['Pclass'].fillna(train['Pclass'].mode()[0],inplace=True)







test['Embarked'].fillna(test['Embarked'].mode()[0],inplace=True)
test['Age'].fillna(test['Age'].mean(),inplace=True)
test['Sex'].fillna(test['Sex'].mode()[0],inplace=True)

test['fam_size'].fillna(test['fam_size'].mode()[0],inplace=True)
test['Fare'].fillna(test['Fare'].mean(),inplace=True)
test['Pclass'].fillna(test['Pclass'].mode()[0],inplace=True)

test['Ticket_Len'].fillna(test['Ticket_Len'].mode()[0],inplace=True)
test['Ticket_Lett'].fillna(test['Ticket_Lett'].mode()[0],inplace=True)
test['Cabin_Letter'].fillna(test['Cabin_Letter'].mode()[0],inplace=True)







test.head()
from sklearn.ensemble import RandomForestClassifier

y = train["Survived"]

features = ["Pclass", "Sex", 'fam_size',"Age","Embarked","Fare","Ticket_Len","Ticket_Lett","Cabin_Letter"]
X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])

Ticket_Lett_5=pd.DataFrame([0]*418)
X_test.insert(14, "Ticket_Lett_5", Ticket_Lett_5,True )
Ticket_Lett_8=pd.DataFrame([0]*418)
X_test.insert(17, "Ticket_Lett_8", Ticket_Lett_8,True )
Cabin_Letter_T=pd.DataFrame([0]*418)
X_test.insert(33, "Cabin_Letter_T", Cabin_Letter_T,True )


model = RandomForestClassifier(n_estimators=700, max_depth=9, random_state=1,oob_score=True)
model.fit(X, y)
print("%.4f" % model.oob_score_)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
y = train["Survived"]

features = ["Pclass", "Sex", 'fam_size',"Age","Embarked","Fare","Ticket_Len","Ticket_Lett","Cabin_Letter"]
X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])

Ticket_Lett_5=pd.DataFrame([0]*418)
X_test.insert(14, "Ticket_Lett_5", Ticket_Lett_5,True )
Ticket_Lett_8=pd.DataFrame([0]*418)
X_test.insert(17, "Ticket_Lett_8", Ticket_Lett_8,True )
Cabin_Letter_T=pd.DataFrame([0]*418)
X_test.insert(33, "Cabin_Letter_T", Cabin_Letter_T,True )

from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
from sklearn.ensemble import RandomForestRegressor

# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X, y)
grid_search.best_params_
best_grid = grid_search.best_estimator_
print(best_grid)
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(max_depth=100, max_features=3, min_samples_leaf=3,
                       min_samples_split=12, oob_score = True)
model.fit(X, y)
print("%.4f" % model.oob_score_)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('my_submission_GSCV.csv', index=False)
print("Your submission was successfully saved!")
#with scaling only
y = train["Survived"]

features =["Pclass", "Sex", 'fam_size',"Age","Embarked","Fare","Ticket_Len","Ticket_Lett","Cabin_Letter"]
X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])

Ticket_Lett_5=pd.DataFrame([0]*418)
X_test.insert(14, "Ticket_Lett_5", Ticket_Lett_5,True )
Ticket_Lett_8=pd.DataFrame([0]*418)
X_test.insert(17, "Ticket_Lett_8", Ticket_Lett_8,True )
Cabin_Letter_T=pd.DataFrame([0]*418)
X_test.insert(33, "Cabin_Letter_T", Cabin_Letter_T,True )


from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
  
X = sc.fit_transform(X) 
X_test = sc.transform(X_test)
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

from sklearn.ensemble import RandomForestRegressor

# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X, y)
grid_search.best_params_
best_grid = grid_search.best_estimator_
print(best_grid)
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(max_depth=80, max_features=3, min_samples_leaf=3,
                       min_samples_split=10, oob_score = True)
model.fit(X, y)
print("%.4f" % model.oob_score_)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('my_submission_GSCV.csv', index=False)
print("Your submission was successfully saved!")
# with Scaling and PCA

y = train["Survived"]

features = ["Pclass", "Sex", 'fam_size',"Age","Embarked","Fare","Ticket_Len","Ticket_Lett","Cabin_Letter"]
X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])


Ticket_Lett_5=pd.DataFrame([0]*418)
X_test.insert(14, "Ticket_Lett_5", Ticket_Lett_5,True )
Ticket_Lett_8=pd.DataFrame([0]*418)
X_test.insert(17, "Ticket_Lett_8", Ticket_Lett_8,True )
Cabin_Letter_T=pd.DataFrame([0]*418)
X_test.insert(33, "Cabin_Letter_T", Cabin_Letter_T,True )

from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
  
X = sc.fit_transform(X) 
X_test = sc.transform(X_test)

from sklearn.decomposition import PCA 
  
pca = PCA(0.95) 
  
X = pca.fit_transform(X) 
X_test = pca.transform(X_test) 
  
explained_variance = pca.explained_variance_ratio_
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

from sklearn.ensemble import RandomForestRegressor

# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X, y)
grid_search.best_params_
best_grid = grid_search.best_estimator_
print(best_grid)
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(max_depth=110, max_features=3, min_samples_leaf=5,
                       min_samples_split=12, n_estimators=200, oob_score = True)
model.fit(X, y)
print("%.4f" % model.oob_score_)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('my_submission_GSCV.csv', index=False)
print("Your submission was successfully saved!")