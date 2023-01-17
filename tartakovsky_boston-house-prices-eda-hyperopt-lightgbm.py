# Re-loads all imports every time the cell is ran. 
%load_ext autoreload
%autoreload 2

import pandas as pd
pd.options.display.float_format = '{:,.5f}'.format

import numpy as np
from time import time
from IPython.display import display

from sklearn.model_selection import cross_validate, learning_curve, train_test_split, RepeatedKFold
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import datasets
from lightgbm import LGBMRegressor

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = datasets.load_boston()
X = pd.DataFrame(data['data'], columns=data['feature_names'])
y = pd.Series(data['target'], name='y')
display(X)
X.describe()
plt.hist(y, bins=50)
n_features = X.shape[1]
n_cols = 4
n_rows = np.ceil(n_features / n_cols).astype(int)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows*3))

for i, col in enumerate(X.columns):
    ax = axes[i//n_cols][i%n_cols]
    ax.hist(X[col], bins=25)

    ax.set_xlabel(X.columns[i])

fig.tight_layout()
n_features = X.shape[1]
pairs = [(X[col], y) for col in X.columns]
n_cols = 4
n_rows = np.ceil(n_features / n_cols).astype(int)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows*3))

for i, pair in enumerate(pairs):
    ax = axes[i//n_cols][i%n_cols]
    ax.scatter(pair[0], pair[1], marker='x')

    ax.set_xlabel(X.columns[i])
    ax.set_ylabel('House price (in $1000s)')

fig.tight_layout()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    random_state=1, test_size=0.2, shuffle=True
)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
rmse = make_scorer(lambda a, b: np.sqrt(mean_squared_error(a, b)), greater_is_better=False)
dummy = DummyRegressor()

# Cross validate dummy classifier to establish no skill performance
dummy_res = cross_validate(dummy, X_train, y_train, scoring=rmse, 
                           return_train_score=True, cv=cv, n_jobs=-1)

dummy_train_score = np.mean(dummy_res['train_score'])
dummy_test_score = np.mean(dummy_res['test_score'])

display(
    f'Train score, dummy: {-dummy_train_score:.2f}',
    f'CV score, dummy: {-dummy_test_score:.2f}', 
)
# Create prediction pipeline
scaler = StandardScaler() 
clf = LinearRegression()

pipeline = make_pipeline(scaler, clf)


# Fit the model, predict train and test sets
res = cross_validate(pipeline, X_train, y_train, scoring=rmse, 
                     return_train_score=True, cv=cv, n_jobs=-1)

train_score = np.mean(res['train_score'])
test_score = np.mean(res['test_score'])

display(
    f'Mean train RMSE: {-train_score:.2f}',
    f'Mean CV RMSE: {-test_score:.2f}', 
)

train_sizes, train_scores, test_scores = learning_curve(
    pipeline, X, y, train_sizes=np.linspace(0.1, 0.8, 20),
    random_state=1, shuffle=True, scoring=rmse
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

fig, ax = plt.subplots(figsize=(10, 5))

sns.lineplot(x=train_sizes, y=train_scores_mean, label='-RMSE: train score', ax=ax)
sns.lineplot(x=train_sizes, y=test_scores_mean, label='-RMSE: CV score', ax=ax)
# Create prediction pipeline
scaler = StandardScaler() 
clf = LGBMRegressor()

pipeline = make_pipeline(scaler, clf)

# Fit the model, predict train and test sets
res = cross_validate(pipeline, X_train, y_train, scoring=rmse, 
                     return_train_score=True, cv=cv, n_jobs=-1)

train_score = np.mean(res['train_score'])
test_score = np.mean(res['test_score'])

display(
    f'Mean train RMSE: {-train_score:.2f}',
    f'Mean CV RMSE: {-test_score:.2f}', 
)

train_sizes, train_scores, test_scores = learning_curve(
    pipeline, X, y, train_sizes=np.linspace(0.1, 0.8, 20),
    random_state=1, shuffle=True, scoring=rmse
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

fig, ax = plt.subplots(figsize=(10, 5))

sns.lineplot(x=train_sizes, y=train_scores_mean, label='-RMSE: train score', ax=ax)
sns.lineplot(x=train_sizes, y=test_scores_mean, label='-RMSE: CV score', ax=ax)
# Create prediction pipeline
scaler = StandardScaler() 
clf = LGBMRegressor(
    reg_lambda=5,
    reg_alpha=2,
)

pipeline = make_pipeline(scaler, clf)

# Fit the model, predict train and test sets
res = cross_validate(pipeline, X_train, y_train, scoring=rmse, 
                     return_train_score=True, cv=cv, n_jobs=-1)

train_score = np.mean(res['train_score'])
test_score = np.mean(res['test_score'])

display(
    f'Mean train RMSE: {-train_score:.2f}',
    f'Mean CV RMSE: {-test_score:.2f}', 
)

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from hyperopt.pyll.base import scope

space = dict(
    learning_rate = hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
    n_estimators = scope.int(hp.qloguniform('n_estimators', np.log(50), np.log(500), np.log(10))),
    max_depth = scope.int(hp.quniform('max_depth', 2, 15, 1)),
)

def objective(params):
        
        clf = make_pipeline(StandardScaler(), LGBMRegressor(**params))
        cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)

        res = cross_validate(clf, X_train, y_train, scoring=rmse, 
                             return_train_score=True, cv=cv, n_jobs=-1)

        train_score = np.mean(res['train_score'])
        test_score = np.mean(res['test_score']) - np.std(res['test_score'])

#         print({ **params, 'loss': test_score})
        result = dict(
            params=params,
            train_loss = -train_score,
            # Hyperopt-required keys
            loss = -test_score,
            status = STATUS_OK,   
        )
        return result
        
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=50, trials=trials)
# results = [trial['result'] for trial in trials]
best
scaler = StandardScaler()

linreg = make_pipeline(scaler, LinearRegression())
lgbm = make_pipeline(scaler, LGBMRegressor())
lgbm_tuned = make_pipeline(scaler, LGBMRegressor(
    learning_rate=best['learning_rate'],
    max_depth=int(best['max_depth']),
    n_estimators=int(best['n_estimators']),
))

linreg.fit(X_train, y_train)
lgbm.fit(X_train, y_train)
lgbm_tuned.fit(X_train, y_train)

print(
    "Linreg:", -rmse(linreg, X_test, y_test),
    "\nDefault LGBM:", -rmse(lgbm, X_test, y_test),
    "\nOptimized LGBM:", -rmse(lgbm_tuned, X_test, y_test)
)
