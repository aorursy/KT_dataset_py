import numpy as np 

import pandas as pd 
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

submission = pd.read_csv('../input/titanic/gender_submission.csv')
train.shape
train.info()
train.head()
train['Survived'].value_counts()
train.columns
train['Cabin'] = train['Cabin'].fillna('None')

train['Embarked'] = train['Embarked'].fillna('None')

train['Have_Cabin'] = np.where(train['Cabin']=='None', 0, 1)
train_dp = train.drop(['PassengerId','Name','Ticket','Cabin','Parch'],axis=1)

train_dp.shape
train_features = train_dp.drop('Survived', axis=1)

train_labels = train_dp['Survived']
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer
num_pipeline =Pipeline([

    ('imputer', SimpleImputer(strategy='median')),

    ('std_scaler',StandardScaler()),

])
num_attribs = list(train_features.select_dtypes(include='number').columns)

cat_attribs = list(train_features.select_dtypes(include='object').columns)
full_pipeline = ColumnTransformer([

    ('num', num_pipeline, num_attribs),

    ('cat', OneHotEncoder(),cat_attribs),

])
cat_attribs
train_prepared = full_pipeline.fit_transform(train_features)
def transform_test(df):

    df['Cabin'] = df['Cabin'].fillna('None')

    df['Embarked'] = df['Embarked'].fillna('None')

    df['Have_Cabin'] = np.where(df['Cabin']=='None', 0, 1)    

    df_dp = df.drop(['PassengerId','Name','Ticket','Cabin','Parch'],axis=1)

    return full_pipeline.transform(df_dp)
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

mod_logit = LogisticRegression(random_state=0)
mod_logit.fit(train_prepared, train_labels)
logit_pred = mod_logit.predict(train_prepared)

accuracy_score(train_labels, logit_pred)
from sklearn.model_selection import cross_val_score

cross_val_score(mod_logit, train_prepared, train_labels, cv=3, scoring="accuracy")
from sklearn.svm import SVC 

mod_svm = SVC(kernel="rbf", gamma=1, C=10)

mod_svm.fit(train_prepared, train_labels)
svm_pred = mod_svm.predict(train_prepared)

accuracy_score(train_labels, svm_pred)
cross_val_score(mod_svm, train_prepared, train_labels, cv=3, scoring="accuracy")
from sklearn.tree import DecisionTreeClassifier

from sklearn import tree
mod_dt = DecisionTreeClassifier(max_depth=3, random_state=42, max_features = 3)

mod_dt.fit(train_prepared, train_labels)
tree.plot_tree(mod_dt);
cross_val_score(mod_dt, train_prepared, train_labels, cv=3, scoring="accuracy")
from sklearn.ensemble import RandomForestClassifier

mod_rf = RandomForestClassifier(random_state=42)

mod_rf.fit(train_prepared, train_labels)
rf_pred = mod_rf.predict(train_prepared)

accuracy_score(train_labels, rf_pred)
cross_val_score(mod_rf, train_prepared, train_labels, cv=3, scoring="accuracy")
from sklearn.ensemble import GradientBoostingClassifier



mod_gbrt = GradientBoostingClassifier(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)

mod_gbrt.fit(train_prepared, train_labels)
cross_val_score(mod_gbrt, train_prepared, train_labels, cv=3, scoring="accuracy")
from sklearn.decomposition import PCA

pca = PCA()

trian_ld = pca.fit_transform(train_prepared)
import matplotlib.pyplot as plt

plt.plot(np.cumsum(pca.explained_variance_ratio_));
import plotly.express as px

df_plot = pd.DataFrame()

df_plot['1d'] = trian_ld[:,0]

df_plot['2d'] = trian_ld[:,1]

df_plot ['label'] = train_labels.astype('category')

px.scatter(df_plot, x='1d', y='2d', color='label')
from sklearn.model_selection import cross_val_predict



y_train_pred = cross_val_predict(mod_rf, train_prepared, train_labels, cv=3)
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
confusion_matrix(train_labels, y_train_pred)
precision_score(train_labels, y_train_pred)
recall_score(train_labels, y_train_pred)
f1_score(train_labels, y_train_pred)
from sklearn.model_selection import GridSearchCV
param_grid_svm = {'C': list(range(1,11)),

                 'gamma': list(range(1,6))}

print(param_grid_svm)
svm_grid = GridSearchCV(mod_svm, param_grid_svm, cv = 3)
svm_grid.fit(train_prepared, train_labels)
svm_grid.best_params_
final_model_svm = svm_grid.best_estimator_
cross_val_score(final_model_svm, train_prepared, train_labels, cv=3, scoring="accuracy")
n_estimators = [int(x) for x in np.linspace(start = 30, stop = 120, num = 10)]

max_features = ['auto', 'sqrt']

bootstrap = [True, False]



# Create the random grid

param_grid_rf = {'n_estimators': n_estimators,

               'max_features': max_features,

               'bootstrap': bootstrap}

print(param_grid_rf)
rf_grid = GridSearchCV(mod_rf, param_grid_rf, cv = 3)
rf_grid.fit(train_prepared, train_labels)
rf_grid.best_params_
final_model_rf = rf_grid.best_estimator_
cross_val_score(final_model_rf, train_prepared, train_labels, cv=3, scoring="accuracy")
from xgboost import XGBClassifier
mod_xg = XGBClassifier()

mod_xg.fit(train_prepared, train_labels)
cross_val_score(mod_xg, train_prepared, train_labels, cv=3, scoring="accuracy")
test.head()
test_prepared = transform_test(test)
test_pred_rf = final_model_rf.predict(test_prepared)

test_pred_dt = mod_dt.predict(test_prepared)

test_pred_svm = final_model_svm.predict(test_prepared)

test_pred_logit = mod_logit.predict(test_prepared)

test_pred_gbrt = mod_gbrt.predict(test_prepared)

test_pred_xg = mod_xg.predict(test_prepared)
test_pred_rf.shape
result = pd.DataFrame()

result['rf'] = test_pred_rf

result['dt'] = test_pred_dt

result['svm'] = test_pred_svm

result['logit'] = test_pred_logit

result['gbrt'] = test_pred_gbrt

result['xg'] = test_pred_xg

result['average_score'] = (result['rf'] + result['dt'] + result['svm'] + result['logit'] + result['xg'])/5

result['average_result'] = (result['average_score'] >= 0.6).apply(int)
submission['Survived'] = result['average_result']
submission.to_csv('submission.csv',index=False)