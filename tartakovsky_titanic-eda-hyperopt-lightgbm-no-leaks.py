# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas_profiling import ProfileReport # library for automatic EDA
from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display # display from IPython.display

from lightgbm import LGBMClassifier

from sklearn.model_selection import cross_validate, learning_curve, train_test_split, RepeatedStratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, FunctionTransformer, PolynomialFeatures
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing the data and displaying some rows
df = pd.read_csv("/kaggle/input/titanic/train.csv")
y = df['Survived'].copy()
X = df.drop(['Survived'], axis=1).copy()
display(df)
report = ProfileReport(df)
display(report)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
# Drop irrelevant features
to_be_removed = ['PassengerId', 'Name', 'Ticket', 'Cabin']
numerical_features = ((X.dtypes == 'float') | (X.dtypes == 'int')) & ~(X.columns.isin(to_be_removed))
categorical_features = ~(numerical_features) & ~(X.columns.isin(to_be_removed))

preprocessor = ColumnTransformer(
    remainder = 'passthrough',
    transformers=[
        (
            'numerical', 
            make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), 
            numerical_features
        ),
        (
            'categorical', 
            make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown="ignore")), 
            categorical_features
        ),
        ('remove', 'drop', to_be_removed),
    ]
)

columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Sex_female', 'Embarked_C', 'Embarked_Q', 'Embarked_S']

display(X, pd.DataFrame(preprocessor.fit_transform(X), columns=columns))
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from hyperopt.pyll.base import scope

space = dict(
    learning_rate = hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
    n_estimators = scope.int(hp.qloguniform('n_estimators', np.log(50), np.log(500), np.log(10))),
    max_depth = scope.int(hp.quniform('max_depth', 2, 15, 1)),
)

def objective(params):
        pipeline = make_pipeline(preprocessor, LGBMClassifier(boosting='gbdt', **params))
        res = cross_validate(pipeline, X, y, scoring='accuracy', return_train_score=True, cv=cv, n_jobs=-1)

        train_score = np.mean(res['train_score'])
        cv_score = np.mean(res['test_score']) - np.std(res['test_score'])
        
        result = dict(
            params=params,
            train_loss = -train_score,
            # Hyperopt-required keys
            loss = -cv_score,
            status = STATUS_OK,   
        )
        return result
        
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=50, trials=trials)
best
clf = LGBMClassifier(
    boosting='gbdt', 
    learning_rate=0.019,
    max_depth=15,
    n_estimators=223,
)
pipeline = make_pipeline(preprocessor, clf)

res = cross_validate(pipeline, X, y, scoring='accuracy', return_train_score=True, cv=cv, n_jobs=-1)

train_score = np.mean(res['train_score'])
cv_score = np.mean(res['test_score'])

display(
    f'Mean train accuracy: {train_score:.3f}',
    f'Mean СV accuracy: {cv_score:.3f}', 
)
pipeline.fit(X, y)

holdout = pd.read_csv("/kaggle/input/titanic/test.csv")

holdout['Survived'] = pipeline.predict(holdout)

submission = holdout[['PassengerId', 'Survived']]

submission.to_csv('submission.csv', index=False)
