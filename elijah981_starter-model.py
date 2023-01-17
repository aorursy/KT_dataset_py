import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

import seaborn as sns # data visualization advanced
df_train = pd.read_csv('/kaggle/input/dataanalyticscoaching/train.csv')

df_test = pd.read_csv('/kaggle/input/dataanalyticscoaching/test.csv')
df_train.head()
df_test.head()
df_train.describe()
df_train.isnull().sum().sum()
df_test.isnull().sum().sum()
df_train['target'].value_counts()
cat_feats = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
fig, ax = plt.subplots(2, 4, figsize=(20,8))

for n, feat in enumerate(cat_feats):

    sns.countplot(x=feat, hue='target', data=df_train, ax=ax[n//4][n%4])
num_feats = list(set(df_train.columns) - set(cat_feats) - {'id', 'target'})

print(num_feats)
fig, ax = plt.subplots(1, 5, figsize=(20,4))

for n, feat in enumerate(num_feats):

    sns.distplot(df_train[df_train['target']==0][feat], ax=ax[n], label='0')

    sns.distplot(df_train[df_train['target']==1][feat], ax=ax[n], label='1')

    ax[n].legend()
fig, ax = plt.subplots(1,1, figsize=(10,10))

sns.heatmap(df_train.corr(), 

            annot=True,

            fmt='.2f',

            cmap=sns.diverging_palette(240, 10, n=25), 

            cbar=False,

            square=True, ax=ax)
X_train = df_train[cat_feats + num_feats]

y_train = df_train['target']
X_train.shape, y_train.shape
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import StandardScaler
class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        return X[self.attribute_names].values
num_pipeline = Pipeline([

                            ('selector', DataFrameSelector(num_feats)),

                            ('std_scaler', StandardScaler())

                        ])
cat_pipeline = Pipeline([

                            ('selector', DataFrameSelector(cat_feats)),

                        ])
num_pipeline.fit_transform(X_train).shape
cat_pipeline.fit_transform(X_train).shape
full_pipeline = FeatureUnion(transformer_list=[

                                ('num_pipeline', num_pipeline),

                                ('cat_pipeline', cat_pipeline)

                            ])
full_pipeline.fit_transform(X_train).shape
X_prep_train = full_pipeline.fit_transform(X_train)
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
LR_clf = LogisticRegression()

LR_clf.fit(X_prep_train, y_train)



y_pred = LR_clf.predict(X_prep_train)

print(classification_report(y_train, y_pred))
from sklearn.model_selection import cross_val_score, KFold
kfold = KFold(n_splits=5)
scores = cross_val_score(LR_clf, X_prep_train, y_train, scoring='accuracy', cv=kfold)

print('Scores:',scores)

print('Mean:',np.mean(scores))

print('Std:',np.std(scores))
from sklearn.model_selection import GridSearchCV
param_grid = {

                'C':[0.01, 0.03, 0.1, 0.3, 1, 3, 10],

                'max_iter':[50, 100, 150],

             }

LR_clf = LogisticRegression()

grid_search = GridSearchCV(LR_clf, param_grid, cv=kfold, scoring='accuracy', verbose=1)



grid_search.fit(X_prep_train, y_train)



print(grid_search.best_params_)



LR_clf = grid_search.best_estimator_

scores = cross_val_score(LR_clf, X_prep_train, y_train, scoring='accuracy', cv=kfold)



LR_clf.fit(X_prep_train, y_train)

y_pred = LR_clf.predict(X_prep_train)

acc = accuracy_score(y_train, y_pred)



df_comp = pd.DataFrame([['Logistic Regression', np.round(acc,2), np.round(np.mean(scores),2)]], 

                       columns=['Algo', 'Train_acc', 'Val_acc'])

df_comp
param_grid = {

                'n_neighbors': range(1, 25, 5),

                'weights':['uniform', 'distance'],

                'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],

             }

KNN_clf = KNeighborsClassifier()

grid_search = GridSearchCV(KNN_clf, param_grid, cv=kfold, scoring='accuracy', verbose=1)



grid_search.fit(X_prep_train, y_train)



print(grid_search.best_params_)



KNN_clf = grid_search.best_estimator_

scores = cross_val_score(KNN_clf, X_prep_train, y_train, scoring='accuracy', cv=kfold)



KNN_clf.fit(X_prep_train, y_train)

y_pred = KNN_clf.predict(X_prep_train)

acc = accuracy_score(y_train, y_pred)



temp = pd.DataFrame([['KNN', np.round(acc,2), np.round(np.mean(scores),2)]], 

                       columns=['Algo', 'Train_acc', 'Val_acc'])

df_comp = df_comp.append(temp,ignore_index=True)

df_comp
param_grid = {

             }

NB_clf = GaussianNB()

grid_search = GridSearchCV(NB_clf, param_grid, cv=kfold, scoring='accuracy')



grid_search.fit(X_prep_train, y_train)



print(grid_search.best_params_)



NB_clf = grid_search.best_estimator_

scores = cross_val_score(NB_clf, X_prep_train, y_train, scoring='accuracy', cv=kfold)



NB_clf.fit(X_prep_train, y_train)

y_pred = NB_clf.predict(X_prep_train)

acc = accuracy_score(y_train, y_pred)



temp = pd.DataFrame([['Naive Bayes', np.round(acc,2), np.round(np.mean(scores),2)]], 

                       columns=['Algo', 'Train_acc', 'Val_acc'])

df_comp = df_comp.append(temp,ignore_index=True)

df_comp
param_grid = {

                'criterion':['gini', 'entropy'],

                'splitter':['best','random'],

                'max_depth':[None] + list(range(1,20, 2)),

                'max_features':[None, 'auto', 'sqrt', 'log2'],

                'ccp_alpha':[0.0, 0.01, 0.03, 0.1]

             }

DT_clf = DecisionTreeClassifier()

grid_search = GridSearchCV(DT_clf, param_grid, cv=kfold, scoring='accuracy', verbose=1)



grid_search.fit(X_prep_train, y_train)



print(grid_search.best_params_)



DT_clf = grid_search.best_estimator_

scores = cross_val_score(DT_clf, X_prep_train, y_train, scoring='accuracy', cv=kfold)



DT_clf.fit(X_prep_train, y_train)

y_pred = DT_clf.predict(X_prep_train)

acc = accuracy_score(y_train, y_pred)



temp = pd.DataFrame([['Decision Tree', np.round(acc,2), np.round(np.mean(scores),2)]], 

                       columns=['Algo', 'Train_acc', 'Val_acc'])

df_comp = df_comp.append(temp,ignore_index=True)

df_comp
param_grid = {

                'C':[0.01, 0.03, 0.1, 0.3, 1, 3, 10],

                'kernel':['linear', 'poly', 'rbf', 'sigmoid'],

                'degree':list(range(1,5)),

             }

SVM_clf = SVC()

grid_search = GridSearchCV(SVM_clf, param_grid, cv=kfold, scoring='accuracy', verbose=1)



grid_search.fit(X_prep_train, y_train)



print(grid_search.best_params_)



SVM_clf = grid_search.best_estimator_

scores = cross_val_score(SVM_clf, X_prep_train, y_train, scoring='accuracy', cv=kfold)



SVM_clf.fit(X_prep_train, y_train)

y_pred = SVM_clf.predict(X_prep_train)

acc = accuracy_score(y_train, y_pred)



temp = pd.DataFrame([['SVM', np.round(acc,2), np.round(np.mean(scores),2)]], 

                       columns=['Algo', 'Train_acc', 'Val_acc'])

df_comp = df_comp.append(temp,ignore_index=True)

df_comp
param_grid = {

                'n_estimators':range(50,200,50),

                'criterion':['gini', 'entropy'],

                'bootstrap':[True, False],

                'max_features':['auto', 'sqrt', 'log2'],

                'ccp_alpha':[0.01, 0.03, 0.1]

             }

RF_clf = RandomForestClassifier()

grid_search = GridSearchCV(RF_clf, param_grid, cv=kfold, scoring='accuracy', verbose=1, n_jobs=-1)



grid_search.fit(X_prep_train, y_train)



print(grid_search.best_params_)



RF_clf = grid_search.best_estimator_

scores = cross_val_score(RF_clf, X_prep_train, y_train, scoring='accuracy', cv=kfold)



RF_clf.fit(X_prep_train, y_train)

y_pred = RF_clf.predict(X_prep_train)

acc = accuracy_score(y_train, y_pred)



temp = pd.DataFrame([['Random Forest', np.round(acc,2), np.round(np.mean(scores),2)]], 

                       columns=['Algo', 'Train_acc', 'Val_acc'])

df_comp = df_comp.append(temp,ignore_index=True)

df_comp
param_grid = {

                'loss':['deviance', 'exponential'],

                'learning_rate':[0.01, 0.03, 0.1, 0.3, 1],

                'n_estimators':list(range(50,200,50)),

                'criterion':['friedman_mse', 'mse', 'mae'],

                'ccp_alpha':[0.01, 0.03, 0.1]

             }

GB_clf = GradientBoostingClassifier()

grid_search = GridSearchCV(GB_clf, param_grid, cv=kfold, scoring='accuracy', verbose=1, n_jobs=-1)



grid_search.fit(X_prep_train, y_train)



print(grid_search.best_params_)



GB_clf = grid_search.best_estimator_

scores = cross_val_score(GB_clf, X_prep_train, y_train, scoring='accuracy', cv=kfold)



GB_clf.fit(X_prep_train, y_train)

y_pred = GB_clf.predict(X_prep_train)

acc = accuracy_score(y_train, y_pred)



temp = pd.DataFrame([['Grad. Boost.', np.round(acc,2), np.round(np.mean(scores),2)]], 

                       columns=['Algo', 'Train_acc', 'Val_acc'])

df_comp = df_comp.append(temp,ignore_index=True)

df_comp
df_test
X_test = df_test[cat_feats + num_feats]

X_prep_test = full_pipeline.fit_transform(X_test)
y_pred_test = SVM_clf.predict(X_prep_test)
submission = pd.DataFrame({

        "id": df_test['id'],

        "target": y_pred_test

    })



submission.to_csv('submission.csv', index=False)