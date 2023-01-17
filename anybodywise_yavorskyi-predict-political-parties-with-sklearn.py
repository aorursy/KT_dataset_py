import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_colwidth', None)
df = pd.read_csv('../input/mse-bb-2-ss2020-mle-congressional-voting/CongressionalVotingID.shuf.lrn.csv')
df.head()
df.drop_duplicates(['class']).groupby('class').head()
df['class'].value_counts()
df.isnull().sum().sum()
X = df.drop(columns=['ID', 'class'])
y = df['class'].values
dataset = (X, y)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron

pipe = Pipeline([
        ('labeler', OneHotEncoder()),
        ('estimator', LogisticRegression())
        ])

param_grid_exhaustive = [
        {
         'estimator': [LogisticRegression()],
         'estimator__penalty': ['l1', 'l2', 'elasticnet', 'none'],
         'estimator__C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
         'estimator__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
         'estimator__class_weight': ['balanced', None],
        },
        {
         'estimator': [LinearSVC()],
         'estimator__penalty': ['l1', 'l2'],
         'estimator__loss': ['hinge', 'squared_hinge'],
         'estimator__C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
         'estimator__multi_class': ['ovr', 'crammer_singer'],
         'estimator__class_weight': ['balanced', None],
        },
        {
         'estimator': [RandomForestClassifier()],
         'estimator__criterion': ['gini', 'entropy'],
         'estimator__max_depth': [1, 2, 3, 4, 5, 7, 9, None],
         'estimator__ccp_alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
         'estimator__class_weight': ['balanced', None],
         'estimator__n_estimators': [10, 50, 100, 200],
        },
        {
         'estimator': [Perceptron()],
         'estimator__penalty': ['l1', 'l2', 'elasticnet', None],
         'estimator__alpha': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
         'estimator__tol': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
         'estimator__class_weight': ['balanced', None],
        }
]

param_grid = [
        {
         'estimator': [LogisticRegression()],
         'estimator__penalty': ['l1'],
         'estimator__C': [0.1],
         'estimator__solver': ['liblinear'],
         'estimator__class_weight': ['balanced'],
        },
]
from sklearn.model_selection import GridSearchCV

X, y = dataset

model = GridSearchCV(pipe, param_grid=param_grid, n_jobs=os.cpu_count(), verbose=10)
model.fit(X, y)

pd.DataFrame.from_dict(model.cv_results_)
import csv

df_pred = pd.read_csv('../input/mse-bb-2-ss2020-mle-congressional-voting/CongressionalVotingID.shuf.tes.csv')

X_pred = df_pred['ID']
y_pred = model.predict(df_pred.drop(columns=['ID']))

df_pred = pd.DataFrame({'ID': X_pred, '"Class"': y_pred})
df_pred.to_csv('../predictions.csv', index=False, quoting=csv.QUOTE_NONE)

df_pred.head()