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
train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
print(model.feature_importances_)
#missingがあるfeature Age を追加
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

y = train_data["Survived"]

features = ["Pclass", "Sex", "Age", "SibSp", "Parch"]

def conv(df):
    my_imputer = SimpleImputer()
    dummied = pd.get_dummies(df)
    imputed = my_imputer.fit_transform(dummied)
    return imputed

X = conv(train_data[features])
X_test = conv(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission2.csv', index=False)
print("Your submission was successfully saved!")

print(model.feature_importances_)
#featureをさらに追加 (Fare)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

y = train_data["Survived"]

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

def conv(df):
    my_imputer = SimpleImputer()
    dummied = pd.get_dummies(df)
    imputed = my_imputer.fit_transform(dummied)
    return imputed

X = conv(train_data[features])
X_test = conv(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission3.csv', index=False)
print("Your submission was successfully saved!")

print(model.feature_importances_)
# SVM を試してみる
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

y = train_data["Survived"]

features = ["Pclass", "Sex", "Age", "SibSp", "Parch"]

def conv(df):
    my_imputer = SimpleImputer()
    dummied = pd.get_dummies(df)
    imputed = my_imputer.fit_transform(dummied)
    return imputed

X = conv(train_data[features])
X_test = conv(test_data[features])
model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
model.fit(X, y)

predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission4.csv', index=False)
print("Your submission was successfully saved!")
# scaler の効果を調べる
# random forest にも scaler 付けてみたが大して変わらなかった
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

y = train_data["Survived"]

features = ["Pclass", "Sex", "Age", "SibSp", "Parch"]

def conv(df):
    my_imputer = SimpleImputer()
    dummied = pd.get_dummies(df)
    imputed = my_imputer.fit_transform(dummied)
    return imputed

X = conv(train_data[features])
X_test = conv(test_data[features])
model = SVC(gamma='auto')
model.fit(X, y)

predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission5.csv', index=False)
print("Your submission was successfully saved!")
from sklearn.utils.fixes import loguniform
print(loguniform(1e0, 1e3))
# randomized search!
# https://scikit-learn.org/stable/modules/grid_search.html
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import RandomizedSearchCV

y = train_data["Survived"]

features = ["Pclass", "Sex", "Age", "SibSp", "Parch"]

def conv(df):
    my_imputer = SimpleImputer()
    dummied = pd.get_dummies(df)
    imputed = my_imputer.fit_transform(dummied)
    return imputed

X = conv(train_data[features])
X_test = conv(test_data[features])

# pipeline & grid search
# https://stackoverflow.com/questions/43366561/use-sklearns-gridsearchcv-with-a-pipeline-preprocessing-just-once/43366811
param_grid = {
    'C': loguniform(1e0, 1e3),
    'gamma': loguniform(1e-4, 1e-3),
    'kernel': ['rbf'],
    'class_weight':['balanced', None]
}
model = make_pipeline(StandardScaler(), RandomizedSearchCV(SVC(), param_grid))
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission6.csv', index=False)
print("Your submission was successfully saved!")
