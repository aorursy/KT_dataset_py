%%time
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
%%time
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()

%%time
X = df.drop('Outcome', axis=1)
X = StandardScaler().fit_transform(X)
y = df['Outcome']
%%time
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)
%%time
model = SVC()

parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
%%time
grid = GridSearchCV(estimator=model, param_grid=parameters, cv=5)
grid.fit(X_train, y_train)
%%time
roc_auc = np.around(np.mean(cross_val_score(grid, X_test, y_test, cv=5, scoring='roc_auc')), decimals=4)
print('Score: {}'.format(roc_auc))
%%time
model = RandomForestClassifier(n_estimators=1000)
model.fit(X_train, y_train)
predictions = cross_val_predict(model, X_test, y_test, cv=5)
print(classification_report(y_test, predictions))
%%time
print(confusion_matrix(y_test, predictions))
%%time
score = np.mean(cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc'))
np.around(score, decimals=4)
%%time
df_diabetes = pd.read_csv('../input/diabetes/diabetes.csv')
dfMerged = pd.merge(df, df_diabetes)
dfMerged.shape
X = dfMerged.drop('Outcome', axis=1)
X = StandardScaler().fit_transform(X)
y = dfMerged['Outcome']
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
score = np.mean(cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc'))
np.around(score, decimals=4)
df_liverpatient = pd.read_csv('../input/indian-liver-patient-records/indian_liver_patient.csv')
df_liverpatient
pd.get_dummies(df_liverpatient['Gender'], prefix = 'Gender').head()
df_liverpatient = pd.concat([df_liverpatient,pd.get_dummies(df_liverpatient['Gender'], prefix = 'Gender')], axis=1)
df_liverpatient["Albumin_and_Globulin_Ratio"] = df_liverpatient.Albumin_and_Globulin_Ratio.fillna(df_liverpatient['Albumin_and_Globulin_Ratio'].mean())

X =df_liverpatient.drop(['Gender','Dataset'], axis=1)


X = StandardScaler().fit_transform(X)
y = df_liverpatient['Dataset']
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
score = np.mean(cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc'))
np.around(score, decimals=4)