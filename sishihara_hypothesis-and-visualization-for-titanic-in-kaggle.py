%matplotlib inline
import numpy as np
import pandas as pd
import re as re
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
data = train.append(test) # The entire data: train + test.
train['Sex'].replace(['male','female'],[0,1], inplace=True)
test['Sex'].replace(['male','female'],[0,1], inplace=True)
data['Embarked'].fillna(('S'), inplace=True)
data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train['Embarked'] = data['Embarked'][:len(train)]
test['Embarked'] = data['Embarked'][len(train):]
data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
data['Categorical_Fare'] = pd.qcut(data['Fare'], 4, labels=False)

train['Categorical_Fare'] = data['Categorical_Fare'][:len(train)]
test['Categorical_Fare'] = data['Categorical_Fare'][len(train):]
age_avg = data['Age'].mean()
age_std = data['Age'].std()

data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)
data['Categorical_Age'] = pd.cut(data['Age'], 5, labels=False)

train['Categorical_Age'] = data['Categorical_Age'][:len(train)]
test['Categorical_Age'] = data['Categorical_Age'][len(train):]
# Dropping Title feature
def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

data['Title'] = data['Name'].apply(get_title)

data['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare', inplace=True)

data['Title'].replace(['Mlle','Ms','Mme'],['Miss','Miss','Mrs'], inplace=True)

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
data['Title'] = data['Title'].map(title_mapping)
data['Title'].fillna(0, inplace=True)
delete_columns = ['Fare', 'Age', 'Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']
train.drop(delete_columns, axis = 1, inplace = True)
test.drop(delete_columns, axis = 1, inplace = True)
train.head()
X = train.drop('Survived', axis = 1)
y = train['Survived']
X_test = test.copy()
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
X_test = std_scaler.transform(X_test)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

n_neighbors = list(range(5,20,1))
algorithm = ['auto']
weights = ['uniform', 'distance']
leaf_size = list(range(1,50,5))
hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 
               'n_neighbors': n_neighbors}
gd = GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True, 
                cv=10, scoring = "roc_auc", n_jobs=10)
gd.fit(X, y)
print(gd.best_score_)
print(gd.best_estimator_)
gd.best_estimator_.fit(X, y)
y_pred = gd.best_estimator_.predict(test)
temp = pd.DataFrame(pd.read_csv("../input/test.csv")['PassengerId'])
temp['Survived'] = list(map(int, y_pred))
temp.to_csv("submission.csv", index = False)
%matplotlib inline 
import matplotlib.pyplot as plt
import seaborn as sns
data['Family_Size'] = data['Parch'] + data['SibSp'] + 1

train['Family_Size'] = data['Family_Size'][:len(train)]
test['Family_Size'] = data['Family_Size'][len(train):]
sns.countplot(x='Family_Size', data = train, hue = 'Survived')
X = train.drop('Survived', axis = 1)
y = train['Survived']
X_test = test.copy()
std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
X_test = std_scaler.transform(X_test)
n_neighbors = list(range(5,20,1))
algorithm = ['auto']
weights = ['uniform', 'distance']
leaf_size = list(range(1,50,5))
hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 
               'n_neighbors': n_neighbors}
gd = GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True, 
                cv=10, scoring = "roc_auc", n_jobs=10)
gd.fit(X, y)
print(gd.best_score_)
print(gd.best_estimator_)
gd.best_estimator_.fit(X, y)
y_pred = gd.best_estimator_.predict(test)
temp = pd.DataFrame(pd.read_csv("../input/test.csv")['PassengerId'])
temp['Survived'] = list(map(int, y_pred))
temp.to_csv("submission_add_family_size.csv", index = False)
data['IsAlone'] = 0
data.loc[data['Family_Size'] == 1, 'IsAlone'] = 1

train['IsAlone'] = data['IsAlone'][:len(train)]
test['IsAlone'] = data['IsAlone'][len(train):]
X = train.drop('Survived', axis = 1)
y = train['Survived']
X_test = test.copy()
std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
X_test = std_scaler.transform(X_test)
n_neighbors = list(range(5,20,1))
algorithm = ['auto']
weights = ['uniform', 'distance']
leaf_size = list(range(1,50,5))
hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 
               'n_neighbors': n_neighbors}
gd = GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True, 
                cv=10, scoring = "roc_auc", n_jobs=10)
gd.fit(X, y)
print(gd.best_score_)
print(gd.best_estimator_)
gd.best_estimator_.fit(X, y)
y_pred = gd.best_estimator_.predict(test)
temp = pd.DataFrame(pd.read_csv("../input/test.csv")['PassengerId'])
temp['Survived'] = list(map(int, y_pred))
temp.to_csv("submission_add_family_size_and_isalone.csv", index = False)
