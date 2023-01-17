import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer, make_column_transformer

from sklearn.pipeline import Pipeline
train_data = pd.read_csv('/kaggle/input/machine-learning-lab-cas-data-science-fs-20/houses_train.csv', index_col=0)
X_data = train_data.drop(columns='price')

y_data = train_data['price']
X_train, X_dev, y_train, y_dev = train_test_split(X_data, y_data, stratify=X_data['object_type_name'], test_size=0.1)
pipeline = Pipeline([

    ('pre', make_column_transformer((OneHotEncoder(handle_unknown='ignore'), ['zipcode', 'municipality_name', 'object_type_name']), remainder='passthrough')),

    ('clf', GradientBoostingRegressor())

])
from sklearn.model_selection import GridSearchCV
parameters = {

    'clf__n_estimators': [10,50,100,200],

    'clf__max_depth': [2,3,5],

}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)
grid_search.fit(X_train, y_train)
grid_search.cv_results_
grid_search.best_params_
pipeline = grid_search.best_estimator_
y_dev_pred = pipeline.predict(X_dev)
def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mean_absolute_percentage_error(y_dev, y_dev_pred)
pipeline.fit(X_data, y_data)
X_test = pd.read_csv('/kaggle/input/machine-learning-lab-cas-data-science-fs-20/houses_test.csv', index_col=0)
y_test_pred = pipeline.predict(X_test)
X_test_submission = pd.DataFrame(index=X_test.index)
X_test_submission['price'] = y_test_pred
X_test_submission.to_csv('gradient_boosting_submission.csv', header=True, index_label='id')