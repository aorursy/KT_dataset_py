# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head(6)
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head(6)
df = train_data.drop(columns=["Ticket", "Cabin"])
dft = test_data.drop(columns=["Ticket", "Cabin"])
df.head(6)
survivors = df[df.Survived == 1]

women_survivors = df[(df.Survived == 1) & (df.Sex == "female")]
percent_women_survivors = len(women_survivors) / len(survivors)

men_survivors = df[(df.Survived == 1) & (df.Sex == "male")]
percent_men_survivors = len(men_survivors) / len(survivors)

print(percent_women_survivors, percent_men_survivors)
class1_survivors = df[(df.Survived == 1) & (df.Pclass == 1)]
percent_class1_survivors = len(class1_survivors) / len(survivors)

class2_survivors = df[(df.Survived == 1) & (df.Pclass == 2)]
percent_class2_survivors = len(class2_survivors) / len(survivors)

class3_survivors = df[(df.Survived == 1) & (df.Pclass == 3)]
percent_class3_survivors = len(class3_survivors) / len(survivors)

print(percent_class1_survivors, percent_class2_survivors, percent_class3_survivors)
df.isna().sum()
dft.isna().sum()
df2 = df.copy()
df2.Age = df.Age.fillna(df.Age.median())
df2.Embarked = df.Embarked.fillna(df.Embarked.mode())

dft2 = dft.copy()
dft2.Age = dft.Age.fillna(dft.Age.median())
dft2.Fare = dft.Fare.fillna(dft.Fare.mean())

df2.head(6)
df3 = df2.copy()
df3.Sex = df3.Sex.map({"male":0, "female":1})

dft3 = dft2.copy()
dft3.Sex = dft3.Sex.map({"male":0, "female":1})

df3.head(6)
titles = ["Mrs", "Mr", "Miss", "Master"]
def find_title(name):
    for title in titles:
        if title in name:
            return title
    return "Other"

DF3 = pd.concat([df3.drop(columns="Survived"), dft3])
DF3["Title"] = DF3.Name.apply(find_title)
DF3.drop("Name", axis=1, inplace=True)
print(DF3.Title.value_counts())
DF4 = pd.get_dummies(DF3)

df4 = DF4.iloc[:len(df3)]
dft4 = DF4.iloc[len(df3):]

print(df4.shape)
print(dft4.shape)
df4.head(6)
dft4.head(6)
X_train = df4.drop(columns=["PassengerId"])
X_test = dft4.drop(columns=["PassengerId"])
y_train = df3["Survived"]
X_train.head(6)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

X_train_sc = X_train.copy()
X_test_sc = X_test.copy()
numerical_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
sc = StandardScaler()
X_train_sc[numerical_features] = sc.fit_transform(X_train[numerical_features])
X_test_sc[numerical_features] = sc.transform(X_test[numerical_features])
X_test_sc.head(6)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV

skf = StratifiedKFold(n_splits=4, shuffle=True)
log = LogisticRegression(max_iter=1000)

param_grid = [{'penalty': ['l2'], 'C': np.linspace(0, 1, 11)}, 
              {'penalty': ['l1'], 'C': np.linspace(0, 1, 11), 'solver': ['liblinear']}]

log_gs = GridSearchCV(log, param_grid, scoring='accuracy', cv=skf, n_jobs=-1)
log_gs.fit(X_train_sc, y_train)

print(log_gs.best_params_)
print(log_gs.best_score_)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

param_grid = {'n_estimators': [100, 500, 1000], 'criterion': ['gini', 'entropy'], 
              'max_depth': [5, 10, None], 'min_impurity_decrease': np.linspace(0.001, 0.01, 10)}

rfc_gs = GridSearchCV(rfc, param_grid, scoring='accuracy', cv=skf, n_jobs=-1)
rfc_gs.fit(X_train, y_train)

print(rfc_gs.best_params_)
print(rfc_gs.best_score_)
from sklearn.ensemble import RandomForestClassifier

# best params = {'criterion': 'entropy', 'max_depth': 10, 'min_impurity_decrease': 0.001, 'n_estimators': 500}
rfc = RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=10, min_impurity_decrease=0.001)
scores = cross_val_score(rfc, X_train, y_train, cv=skf, n_jobs=-1)

print(scores)
print(scores.mean())
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

dt = DecisionTreeClassifier()
abc = AdaBoostClassifier(dt)

param_grid = {'base_estimator__max_depth': np.arange(1, 6), 
              'n_estimators': [100, 500, 1000], 'learning_rate': np.linspace(0.1, 0.9, 9)}

abc_gs = GridSearchCV(abc, param_grid, scoring='accuracy', cv=skf, n_jobs=-1)
abc_gs.fit(X_train, y_train)

print(abc_gs.best_params_)
print(abc_gs.best_score_)
# best params = {'base_estimator__max_depth': 1, 'learning_rate': 0.9, 'n_estimators': 100}
abc = AdaboostClassifier(learning_rate=0.9)

scores = cross_val_score(rfc, X_train, y_train, cv=skf, n_jobs=-1)

print(scores)
print(scores.mean())
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()

param_grid = {'n_estimators': [100, 500, 1000], 'learning_rate': np.linspace(0.1, 0.9, 9),
              'max_depth': [5, 10, 20], 'min_impurity_decrease': np.linspace(0.001, 0.01, 10)}

gbc_gs = GridSearchCV(gbc, param_grid, scoring='accuracy', cv=skf, n_jobs=-1)
gbc_gs.fit(X_train, y_train)

print(gbc_gs.best_params_)
print(gbc_gs.best_score_)
# best params = {'learning_rate': 0.1, 'max_depth': 5, 'min_impurity_decrease': 0.001, 'n_estimators': 100}
gbc = GradientBoostingClassifier(learning_rate=0.1, max_depth=5, min_impurity_decrease=0.001)
scores = cross_val_score(gbc, X_train, y_train, cv=skf)

print(scores)
print(scores.mean())
from xgboost import XGBClassifier

xgb = XGBClassifier(max_depth=4, learning_rate=0.3, gamma=0.0004, objective='binary:logistic')
scores = cross_val_score(xgb, X_train_sc, y_train, cv=skf)

print(scores)
print(scores.mean())
from sklearn.svm import SVC

svm = SVC()
param_grid = [{'kernel': ['rbf'], 'C': np.linspace(0, 1, 11)}, 
              {'kernel': ['poly'], 'C': np.linspace(0, 1, 11), 'degree': [2, 3, 4]}]

svm_gs = GridSearchCV(svm, param_grid, scoring='accuracy', cv=skf, n_jobs=-1)
svm_gs.fit(X_train_sc, y_train)

print(svm_gs.best_params_)
print(svm_gs.best_score_)
model = svm_gs
x_train = X_train_sc
x_test = X_test_sc

model.fit(x_train, y_train)
predictions = model.predict(x_test)
output = pd.DataFrame({'PassengerId': dft4.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
output.head(10)