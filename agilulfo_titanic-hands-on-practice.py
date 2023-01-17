# Basic libraries

import numpy as np

import pandas as pd



# scikit-learn

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import cross_val_score



# XGBoost

from xgboost import XGBClassifier
titanic_filepath = '../input/titanic/train.csv'

titanic_data = pd.read_csv(titanic_filepath, index_col='PassengerId')



titanic_test_filepath = '../input/titanic/test.csv'

titanic_test_data = pd.read_csv(titanic_test_filepath, index_col='PassengerId')
y = titanic_data['Survived']

X = titanic_data.drop(['Survived'], axis=1)



X_test = titanic_test_data
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

numerical_transformer = SimpleImputer(strategy='median')



categorical_cols = [cname for cname in X.columns if X[cname].dtype == 'object' and X[cname].nunique() < 10]

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor = ColumnTransformer(transformers=[

    ('num', numerical_transformer, numerical_cols),

    ('cat', categorical_transformer, categorical_cols)

])
model = XGBClassifier(n_estimators=200)
pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('model', model)

])
scores = -1 * cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')

print(scores.mean())
pipeline.fit(X, y)

preds_test = pipeline.predict(X_test)

output = pd.DataFrame({

    'PassengerId': X_test.index,

    'Survived': preds_test

})

output.to_csv('submission.csv', index=False)