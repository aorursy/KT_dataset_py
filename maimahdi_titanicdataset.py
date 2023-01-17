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
titanic_train = pd.read_csv('../input/titanic/train.csv', index_col = 'PassengerId')

titanic_test = pd.read_csv('../input/titanic/test.csv', index_col = 'PassengerId')
titanic_train
titanic_train.isna().sum()
len(titanic_train.Name.unique())
titanic_train.Survived.value_counts()
for col in titanic_train.columns:

    print(col, len(titanic_train[col].unique()))
titanic_train.drop(columns=['Name', 'Cabin', 'Ticket'], inplace=True)

titanic_train
age_mean = titanic_train.Age.mean()

titanic_train.Age.fillna(age_mean, inplace=True)

titanic_train.isna().sum()
from sklearn.preprocessing import LabelEncoder





sex_le =  LabelEncoder() 

titanic_train['Sex'] = sex_le.fit_transform(titanic_train['Sex'].astype(str))



embarked_le =  LabelEncoder() 

titanic_train['Embarked'] = embarked_le.fit_transform(titanic_train['Embarked'].astype(str))
titanic_train.Sex
titanic_train.Embarked
target_column = titanic_train.Survived

y = target_column

X = titanic_train.drop(columns=['Survived'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
para = list(range(100, 1001, 100))

print(para)
from sklearn.metrics import classification_report, accuracy_score, f1_score

from xgboost import  XGBClassifier



results = {}

for n in para:

    print('para=', n)

    titanic_model = XGBClassifier(n_estimators=n, learning_rate=0.05, n_jobs=4)

    titanic_model.fit(X_train, y_train, early_stopping_rounds=20,

              eval_set=[(X_test, y_test)], 

             verbose=False)

    preds = titanic_model.predict(X_test)

    accu = accuracy_score(y_true=y_test, y_pred=preds)

    f1 = f1_score(y_true=y_test, y_pred=preds, average='micro')

    print(classification_report(y_true=y_test, y_pred=preds))

    print('--------------------------')

    results[n] = f1
best_para = max(results, key=results.get)

print('best para', best_para)

print('value', results[best_para])
titanic_test
titanic_test.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

titanic_test
titanic_test['Sex'] = sex_le.transform(titanic_test['Sex'].astype(str))

titanic_test['Embarked'] = embarked_le.transform(titanic_test['Embarked'].astype(str))
titanic_test
titanic_final_model = XGBClassifier(n_estimators=best_para, learning_rate=0.05, n_jobs=4)

titanic_final_model.fit(X, y)
titanic_test.isna().sum()
titanic_test.Age.fillna(int(age_mean), inplace=True)

titanic_test.Fare.fillna(int(titanic_train.Fare.mean()), inplace=True)

titanic_train
preds = titanic_final_model.predict(titanic_test)
preds
preds[:5]
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score, f1_score

results = {}

for n in para:

    print('para=', n)

    model = RandomForestClassifier(n_estimators=n)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    accu = accuracy_score(y_true=y_test, y_pred=preds)

    f1 = f1_score(y_true=y_test, y_pred=preds, average='micro')

    print(classification_report(y_true=y_test, y_pred=preds))

    print('--------------------------')

    results[n] = f1
best_para = max(results, key=results.get)

print('best para', best_para)

print('value', results[best_para])
final_model = RandomForestClassifier(n_estimators=best_para)

final_model.fit(X, y)
predictions = final_model.predict(titanic_test)
predictions
test_out = pd.DataFrame({

    'PassengerId': titanic_test.index, 

    'Survived': preds

})

test_out.to_csv('submission.csv', index=False)