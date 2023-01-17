import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer, make_column_transformer

from sklearn.pipeline import Pipeline
train_data = pd.read_csv('/kaggle/input/cas-ds-fs-20/houses_train.csv', index_col=0)
X_train = train_data.drop(columns='price')

y_train = train_data['price']
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, stratify=X_train['object_type_name'], test_size=0.1)
pipeline = Pipeline([

    ('pre', make_column_transformer((OneHotEncoder(handle_unknown='ignore'), ['zipcode', 'municipality_name', 'object_type_name']), remainder='passthrough')),

    ('clf', LinearRegression())

])
pipeline.fit(X_train, y_train)
y_dev_pred = pipeline.predict(X_dev)
def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mean_absolute_percentage_error(y_dev, y_dev_pred)
X_test = pd.read_csv('/kaggle/input/cas-ds-fs-20/houses_train.csv', index_col=0)
y_test_pred = pipeline.predict(X_test)
X_test_submission = pd.DataFrame(index=X_test.index)
X_test_submission['price'] = y_test_pred
X_test_submission.to_csv('lr_submission.csv', header=True, index_label='id')