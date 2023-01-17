import pandas as pd

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

import numpy as np

from sklearn.base import TransformerMixin
train_df = pd.read_csv('../input/train.csv', header=0)

test_df = pd.read_csv('../input/test.csv', header=0)
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



big_X_imputed.head(7)

le = LabelEncoder()

for feature in nonnumeric_columns:

    big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])



big_X_imputed.head(3)
train_X = big_X_imputed[0:train_df.shape[0]].as_matrix()

test_X = big_X_imputed[train_df.shape[0]::].as_matrix()

train_y = train_df['Survived']
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)

predictions = gbm.predict(test_X)
submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],

                            'Survived': predictions })

submission.to_csv("submission.csv", index=False)