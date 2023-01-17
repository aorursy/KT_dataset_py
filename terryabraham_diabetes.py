%%time
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
%%time
model = SVC()

parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
%%time
grid = GridSearchCV(estimator=model, param_grid=parameters, cv=5)
grid.fit(X_train, y_train)
%time
roc_auc = np.around(np.mean(cross_val_score(grid, X_test, y_test, cv=5, scoring='roc_auc')), decimals=4)
print('Score: {}'.format(roc_auc))
%time
model1= RandomForestClassifier(n_estimators=1000)
model1.fit(X_train, y_train)
predictions = cross_val_predict(model1, X_test, y_test, cv=5)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
%time
score1= np.mean(cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc'))
np.around(score1, decimals=4)
%%time
df_diabetes2=pd.read_csv('../input/disease/diabetes_2.csv')
df_diabetes2.head()
%%time
dfmerge=pd.merge(df,df_diabetes2)
dfmerge.shape,dfmerge.head()
%%time
X = dfmerge.drop('Outcome', axis=1)
X = StandardScaler().fit_transform(X)
y = dfmerge['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
%%time
model = SVC()

parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
grid = GridSearchCV(estimator=model, param_grid=parameters, cv=5)
grid.fit(X_train, y_train)
merge_scores = np.around(np.mean(cross_val_score(grid, X_test, y_test, cv=5, scoring='roc_auc')), decimals=4)
print('Score: {}'.format(merge_scores))
%%time
model = RandomForestClassifier(n_estimators=1000)
model.fit(X_train, y_train)
score =np.around(np.mean(cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc')),decimals=4)
print('Score: {}'.format(score))
