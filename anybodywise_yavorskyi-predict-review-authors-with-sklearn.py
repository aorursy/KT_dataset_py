import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_colwidth', None)
df = pd.read_csv('../input/mse-bb-2-ss2020-mle-amazon-reviews/amazon_review_ID.shuf.lrn.csv')
df.shape
df.head()
df.drop_duplicates(['Class']).groupby('Class').head()
df['Class'].value_counts()
df.isnull().sum().sum()
X = df.drop(columns=['ID', 'Class'])
y = df['Class'].values
dataset = (X, y)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron

pipe = Pipeline([
        ('scaler', QuantileTransformer()),
        ('reduce_dim', SelectKBest(f_classif)), 
        ('estimator', LinearSVC())
        ])

scalers = [
        StandardScaler(),
        MinMaxScaler(),
        QuantileTransformer(),
        None
]

param_grid_exhaustive = [
        {
         'scaler': scalers,
         'reduce_dim__k': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
         'estimator': [LogisticRegression()],
         'estimator__penalty': ['l1', 'l2', 'elasticnet', 'none'],
         'estimator__C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
         'estimator__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
         'estimator__class_weight': ['balanced', None],
        },
        {
         'scaler': scalers,
         'reduce_dim__k': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
         'estimator': [LinearSVC()],
         'estimator__penalty': ['l1', 'l2'],
         'estimator__loss': ['hinge', 'squared_hinge'],
         'estimator__C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
         'estimator__multi_class': ['ovr', 'crammer_singer'],
         'estimator__class_weight': ['balanced', None],
        },
        {
         'scaler': scalers,
         'reduce_dim__k': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
         'estimator': [Perceptron()],
         'estimator__penalty': ['l1', 'l2', 'elasticnet', None],
         'estimator__alpha': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
         'estimator__tol': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
         'estimator__class_weight': ['balanced', None],
        }
]

param_grid = [
        {
         'scaler': [QuantileTransformer()],
         'reduce_dim__k': [800],
         'estimator': [LinearSVC()],
         'estimator__penalty': ['l2'],
         'estimator__loss': ['squared_hinge'],
         'estimator__C': [0.1],
         'estimator__multi_class': ['ovr'],
        }
]
from sklearn.model_selection import GridSearchCV

X, y = dataset

model = GridSearchCV(pipe, param_grid=param_grid, n_jobs=os.cpu_count(), verbose=10)
model.fit(X, y)

pd.DataFrame.from_dict(model.cv_results_)
import csv

df_pred = pd.read_csv('../input/mse-bb-2-ss2020-mle-amazon-reviews/amazon_review_ID.shuf.tes.csv')

X_pred = df_pred['ID']
y_pred = model.predict(df_pred.drop(columns=['ID']))

df_pred = pd.DataFrame({'ID': X_pred, '"Class"': y_pred})
df_pred.to_csv('../predictions.csv', index=False, quoting=csv.QUOTE_NONE)

df_pred.head()