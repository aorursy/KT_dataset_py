import numpy as np

import pandas as pd

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

from sklearn.base import TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures

import os

print(os.listdir("../input"))
train_df = pd.read_csv('../input/train.csv', header=0)

train_df.head()
test_df = pd.read_csv('../input/test.csv', header=0)

test_df.head()
class DataFrameImputer(TransformerMixin):

    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]

            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],

            index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.fill)
feature_columns_to_use = ['Pclass','Sex','Age','Fare','Parch']

nonnumeric_columns = ['Sex']



big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])

big_X_imputed = DataFrameImputer().fit_transform(big_X)
le = LabelEncoder()

for feature in nonnumeric_columns:

    big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])
train_X = big_X_imputed[0:train_df.shape[0]].as_matrix()

test_X = big_X_imputed[train_df.shape[0]::].as_matrix()

train_y = train_df['Survived']
steps = [('poly_features', PolynomialFeatures(3, interaction_only=True)),

         ('XGBoost', xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.01))]

gbm = Pipeline(steps=steps)

gbm.fit(train_X, train_y)

print('Accuracy (xgboost-Poly Features (cubic)): {:.4f}'.format(gbm.score(train_X, train_y)))         



predictions = gbm.predict(test_X)
submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],

                           'Survived': predictions })

submission.to_csv("submission.csv", index=False)