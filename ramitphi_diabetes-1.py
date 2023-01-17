import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

df.head()
df.columns
df.info()
sns.countplot(x=df['Outcome'])
X = df.drop('Outcome', axis=1)

X = StandardScaler().fit_transform(X)

y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.25, random_state=0)
model = SVC()



parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],

                     'C': [1, 10, 100, 1000]},

                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
grid = GridSearchCV(estimator=model, param_grid=parameters, cv=5)

grid.fit(X_train, y_train)
roc_auc = np.around(np.mean(cross_val_score(grid, X_test, y_test, cv=5, scoring='roc_auc')), decimals=4)

print('Score: {}'.format(roc_auc))
model = RandomForestClassifier(n_estimators=1000)

model.fit(X_train, y_train)

predictions = cross_val_predict(model, X_test, y_test, cv=5)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
score = np.mean(cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc'))

np.around(score, decimals=4)
df_2 =pd.read_csv('../input/diabetes/diabetes.csv')
df_2.head()
df_2.info()
df_2['Outcome'].value_counts()
X_2 = df_2.drop('Outcome', axis=1)

X_2 = StandardScaler().fit_transform(X_2)

y_2 = df_2['Outcome']
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(

    X_2, y_2, test_size=0.25, random_state=0)
model1 = SVC()



parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],

                     'C': [1, 10, 100, 1000]},

                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
grid = GridSearchCV(estimator=model1, param_grid=parameters, cv=5)

grid.fit(X_train_2, y_train_2)
roc_auc = np.around(np.mean(cross_val_score(grid, X_test_2, y_test_2, cv=5, scoring='roc_auc')), decimals=4)

print('Score: {}'.format(roc_auc))