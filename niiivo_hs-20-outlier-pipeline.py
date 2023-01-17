import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import pandas as pd



from sklearn.compose import make_column_transformer

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.base import BaseEstimator, TransformerMixin
data = pd.read_csv("/kaggle/input/machine-learning-lab-cas-data-science-hs-20/train-data.csv", index_col=0)
train_data, dev_data = train_test_split(data, test_size=0.1, random_state=0)
X_train, y_train = train_data.drop(columns=['G1', 'G2', 'G3']), train_data['G3']

X_dev, y_dev = dev_data.drop(columns=['G1', 'G2', 'G3']), dev_data['G3']
num_features = ['age', 'absences', 'failures', 'studytime', 'Medu', 'Fedu', 'goout', 'absences', 'freetime']

cat_features = ['sex']
class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, columns):

        self.columns = columns



    def fit(self, X, y=None):

        return self



    def transform(self, X, y=None):

        return X[self.columns]

    

class OutlierQuantileRemover(BaseEstimator, TransformerMixin):

    def __init__(self, column, q=0.99):

        self.column = column

        self.q = q

        self.quantile = None



    def fit(self, X, y=None):

        self.quantile = X[self.column].quantile()

        return self



    def transform(self, X, y=None):

        X.loc[X[self.column] > self.quantile, self.column] = self.quantile

        return X



class OutlierThresholdRemover(BaseEstimator, TransformerMixin):

    def __init__(self, column, t):

        self.column = column

        self.t = t



    def fit(self, X, y=None):

        return self



    def transform(self, X, y=None):

        X.loc[X[self.column] > self.t, self.column] = self.t

        return X



pipeline = Pipeline([

    ('selector', FeatureSelector(columns=num_features+cat_features)),

    ('out', OutlierThresholdRemover(column='failures', t=3)),

    ('out2', OutlierQuantileRemover(column='freetime')),

    ('pre', make_column_transformer((OneHotEncoder(handle_unknown='ignore'), cat_features), remainder='passthrough')),

    ('clf', LinearRegression(normalize=True))

])



pipeline.fit(X_train, y_train)
y_pred_train = pipeline.predict(X_train)

y_pred_dev = pipeline.predict(X_dev)



print('train: ', mean_absolute_error(y_train, y_pred_train))

print('dev:  ', mean_absolute_error(y_dev, y_pred_dev))
X_test = pd.read_csv("/kaggle/input/machine-learning-lab-cas-data-science-hs-20/test-data.csv", index_col=0)
y_test_pred = pipeline.predict(X_test)
X_test_submission = pd.DataFrame(index=X_test.index)

X_test_submission['G3'] = y_test_pred

X_test_submission.to_csv('submission_outlier_pipeline.csv', header=True, index_label='id')