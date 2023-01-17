import numpy as np

import pandas as pd

import seaborn as sns



df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')



df_test.head()
print(df_train.shape, df_test.shape)
df_train.describe()
df_train.apply(lambda x: x.isna().sum())

df_test.apply(lambda x: x.isna().sum())

df_train.drop(['Name', 'Ticket', 'Cabin'], inplace=True, axis=1)

df_train = df_train.fillna(method='ffill')



df_test.drop(['Name', 'Ticket', 'Cabin'], inplace=True, axis=1)

df_test = df_test.fillna(method='ffill')



df_train.head()
numeric = ['Pclass', 'Age', 'Fare']

df_num = df_train[numeric]

df_num.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df_num)

scaled = scaler.transform(df_num)

df_num_scaled = pd.DataFrame(scaled)
cat = ['Sex', 'Embarked']

df_cat = df_train[cat]

df_dum = pd.get_dummies(df_cat)
df_dum.apply(lambda x: x.isna().sum())
X = pd.concat([df_num_scaled, df_dum], axis=1, join='inner')

y = df_train['Survived']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
y.shape
from sklearn.svm import SVC

svc = SVC()

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
y_test = y_test.values.reshape(-1, 1)

print(y_test.shape, y_pred.shape)
y_pred = y_pred.reshape(-1, 1)

y_test = y_test.reshape(-1, 1)



svc.score(X_test, y_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
sns.heatmap(cm, cmap='Blues',annot = True, fmt='')
import xgboost as xgb
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

xgb_model.fit(X_train, y_train)



y_pred = xgb_model.predict(X_test)



xgb_model.score(X_test, y_test)
print(confusion_matrix(y_test, y_pred))
X = df_train.drop('Survived', axis=1)

y = df_train['Survived']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_test
passengerId = X_test.PassengerId



X_train = X_train.drop('PassengerId', axis=1)

X_test = X_test.drop('PassengerId', axis=1)
X_train
from catboost import CatBoostClassifier

cat_features = ['Sex', 'Embarked']
clf = CatBoostClassifier(

    iterations=5, 

    learning_rate=0.1, 

#     loss_function='CrossEntropy'

)



clf.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test), verbose=False)
y_pred_clf = clf.predict(X_test)

score = clf.score(X_test, y_test)
passengerId = df_test.PassengerId

df_test = df_test.drop(['PassengerId'], axis=1)

df_test.shape
y_pred_final = clf.predict(df_test)
y_pred_final.shape
df1 = pd.DataFrame({'PassengerId': passengerId, 'Survived': y_pred_final})
df1.to_csv('submission.csv', index=False)
df1.shape