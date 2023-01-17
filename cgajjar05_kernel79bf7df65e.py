# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_train.head()
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

df_test.head()
df_train_filtered = df_train.drop(columns = ['PassengerId','Name','Ticket','Cabin','Embarked'])
df_train_filtered.head()
df_train_filtered.describe()
df_dummy = pd.get_dummies(df_train_filtered, columns=['Pclass','Sex'])
df_dummy.head()
df_dummy_filled = df_dummy.fillna(df_dummy.mean())
Y = df_dummy_filled['Survived']

X = df_dummy_filled.drop(columns='Survived')
Y.head()
X.head()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()

# scaler.fit(X)

# X = scaler.transform(X)
X


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=41)
clf = RandomForestClassifier(n_estimators=100, random_state=5, max_features=3, max_depth=4)
clf.fit(X_train, y_train)
accuracy_score(y_train, clf.predict(X_train))
accuracy_score(y_test, clf.predict(X_test))
confusion_matrix(y_train, clf.predict(X_train))
confusion_matrix(y_test, clf.predict(X_test))
feature_importances = pd.DataFrame(clf.feature_importances_,index = X_train.columns,columns=['importance']).sort_values('importance', ascending=False)
feature_importances
import xgboost as xgb
xgb_model = xgb.XGBClassifier(random_state=42, learning_rate=0.085, max_depth=3)
xgb_model.fit(X_train, y_train)
accuracy_score(y_train, xgb_model.predict(X_train))
accuracy_score(y_test, xgb_model.predict(X_test))
df_test_filtered = df_test.drop(columns = ['PassengerId','Name','Ticket','Cabin','Embarked'])
pass_id = df_test['PassengerId']
df_dummy_test = pd.get_dummies(df_test_filtered, columns=['Pclass','Sex'])
df_dummy_filled_test = df_dummy_test.fillna(df_dummy_test.mean())
pred = xgb_model.predict(df_dummy_filled_test)
df_answer = pd.DataFrame()
df_answer['PassengerId'] = pass_id

df_answer['Survived'] = pred
df_answer.to_csv('titanic_answer.csv', index=False)