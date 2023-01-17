import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.datasets import load_breast_cancer

import warnings

warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')

sns.set_style("white")
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA



from sklearn.model_selection import GridSearchCV



from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier



from sklearn.metrics import accuracy_score, classification_report
dataset = load_breast_cancer()

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

df['target'] = dataset.target

df.head()
# check for null values

df.isna().sum()
X = df.drop('target', axis='columns')

y = df.target
print(X.shape)

print(y.shape)
X.info()
X.describe()
#Check distribution of classes in target

sns.countplot(df['target'],label='count')
y.value_counts()
# distribution of every feature 

for column in X:

    plt.figure()

    column_data = X[column]

    sns.kdeplot(column_data[y == 0], label='B')

    sns.kdeplot(column_data[y == 1], label='M')

    plt.xlabel(column)

    plt.ylabel('Frequency')

    plt.title('Distribution of {}'.format(column))
# Calculate and visualise correlations between features

plt.figure(figsize=(20, 16))

sns.heatmap(X.corr(), annot=True, cmap="YlGnBu")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
pipe = Pipeline(steps=[

    ('scaler', StandardScaler()),

    ('pca', PCA(n_components=2))

])

X_pca = pipe.fit_transform(X_train, y_train)

sns.scatterplot(X_pca[:,0], X_pca[:,1], hue=y_train.map({0:'M', 1:'B'}))

plt.xlabel('Component 1')

plt.ylabel('Component 2')

plt.title('First two principal components of dataset')
knn_pipe = Pipeline(steps=[

    ('scaler', StandardScaler()),

    ('pca', PCA()),

    ('knn', KNeighborsClassifier())

])

param_grid = {

    'pca__n_components': np.arange(1, X_train.shape[1]+1),

    'knn__n_neighbors': np.arange(1, X_train.shape[1], 2)

}

knn_model = GridSearchCV(knn_pipe, param_grid=param_grid, verbose=1, n_jobs=-1)

knn_model.fit(X_train, y_train)



print('Best params: {}'.format(knn_model.best_params_))

print('Training Score: {}'.format(knn_model.score(X_train, y_train)))

print('CV Score: {}'.format(knn_model.best_score_))

print('Test Score: {}'.format(knn_model.score(X_test, y_test)))
gnb_pipe = Pipeline(steps=[

    ('scaler', StandardScaler()),

    ('pca', PCA()),

    ('gnb', GaussianNB())

])

param_grid = {

    'pca__n_components': np.arange(1, X_train.shape[1]+1)

}

gnb_model = GridSearchCV(gnb_pipe, param_grid=param_grid, verbose=1, n_jobs=-1)

gnb_model.fit(X_train, y_train)

print('Best params: {}'.format(gnb_model.best_params_))

print('Training Score: {}'.format(gnb_model.score(X_train, y_train)))

print('CV Score: {}'.format(gnb_model.best_score_))

print('Test Score: {}'.format(gnb_model.score(X_test, y_test)))
lgr_pipe = Pipeline(steps=[

    ('scaler', StandardScaler()),

    ('pca', PCA()),

    ('lgr', LogisticRegression())

])

param_grid = {

    'pca__n_components': np.arange(1, X_train.shape[1]//3),

    'lgr__C': np.logspace(0, 1, 10)

}

lgr_model = GridSearchCV(lgr_pipe, param_grid=param_grid, verbose=1, n_jobs=-1)

lgr_model.fit(X_train, y_train)

print('Best params: {}'.format(lgr_model.best_params_))

print('Training Score: {}'.format(lgr_model.score(X_train, y_train)))

print('CV Score: {}'.format(lgr_model.best_score_))

print('Test Score: {}'.format(lgr_model.score(X_test, y_test)))
rdf_pipe = Pipeline(steps=[

    ('scaler', StandardScaler()),

    ('rdf', RandomForestClassifier())

])

param_grid = {

    'rdf__n_estimators': np.arange(200, 1001, 200),

    'rdf__max_depth': np.arange(1,4),

}

rdf_model = GridSearchCV(rdf_pipe, param_grid=param_grid, verbose=1, n_jobs=-1)

rdf_model.fit(X_train, y_train)

print('Best params: {}'.format(rdf_model.best_params_))

print('Training Score: {}'.format(rdf_model.score(X_train, y_train)))

print('CV Score: {}'.format(rdf_model.best_score_))

print('Test Score: {}'.format(rdf_model.score(X_test, y_test)))
svc_pipe = Pipeline(steps=[

    ('scaler', StandardScaler()),

    ('pca', PCA()),

    ('svc', SVC())

])

param_grid = {

    'pca__n_components': np.arange(1, X_train.shape[1]//3),

    'svc__C': np.logspace(0, 3, 10),

    'svc__kernel': ['rbf'],

    'svc__gamma': np.logspace(-4, -3, 10)

}

svc_model = GridSearchCV(svc_pipe, param_grid=param_grid, verbose=1, n_jobs=-1)

svc_model.fit(X_train, y_train)

print('Best params: {}'.format(svc_model.best_params_))

print('Training Score: {}'.format(svc_model.score(X_train, y_train)))

print('CV Score: {}'.format(svc_model.best_score_))

print('Test Score: {}'.format(svc_model.score(X_test, y_test)))
xgb_pipe = Pipeline(steps=[

    ('scaler', StandardScaler()),

#     ('pca', PCA()),

    ('xgb', XGBClassifier())

])

param_grid = {

#     'pca__n_components': np.arange(1, X_train.shape[1]//3),

    'xgb__n_estimators': [100],

    'xgb__learning_rate': np.logspace(-3, 0, 10),

    'xgb__max_depth': np.arange(1, 6),

    'xgb__gamma': np.arange(0, 1.0, 0.1),

    'xgb__reg_lambda': np.logspace(-3, 3, 10)

}

xgb_model = GridSearchCV(xgb_pipe, param_grid=param_grid, verbose=1, n_jobs=-1)

xgb_model.fit(X_train, y_train)

print('Best params: {}'.format(xgb_model.best_params_))

print('Training Score: {}'.format(xgb_model.score(X_train, y_train)))

print('CV Score: {}'.format(xgb_model.best_score_))

print('Test Score: {}'.format(xgb_model.score(X_test, y_test)))


%%time

models = {

    'KNN': knn_model,

    'GaussianNB': gnb_model,

    'LogisticRegression': lgr_model,

    'RandomForests': rdf_model,

    'SVC': svc_model,

    'XGBoost': xgb_model

}

y_stacked = pd.DataFrame({model_name: model.predict(X_train) for model_name, model in models.items()})

y_stacked_train, y_stacked_test, y_train_train, y_train_test = train_test_split(y_stacked, y_train, 

                                                                              random_state=0, stratify=y_train)

param_grid = {

    'C': np.logspace(0, 3, 10),

    'kernel': ['rbf'],

    'gamma': np.logspace(-3, 3, 10)

}

stacked_model = GridSearchCV(SVC(), param_grid=param_grid, verbose=1, n_jobs=-1)

stacked_model.fit(y_stacked_train, y_train_train)

print('Best params: {}'.format(stacked_model.best_params_))

print('Training Score: {}'.format(stacked_model.score(y_stacked_train, y_train_train)))

print('CV Score: {}'.format(stacked_model.best_score_))

print('Test Score: {}'.format(stacked_model.score(y_stacked_test, y_train_test)))
y_stacked = pd.DataFrame({model_name: model.predict(X_test) for model_name, model in models.items()})

y_pred = stacked_model.predict(y_stacked)

print('Overall Accuracy Score: {:.2%}'.format(accuracy_score(y_test, y_pred)))

print('Classification report:')

print(classification_report(y_test, y_pred))