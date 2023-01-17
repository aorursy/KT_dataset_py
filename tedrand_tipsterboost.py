import pandas as pd

import matplotlib.pyplot as plt

import xgboost as xgb

from sklearn.model_selection import train_test_split, GridSearchCV # split data

import seaborn as sns; # plotting

%matplotlib inline
# !!!Change encoding to latin1

df = pd.read_csv('../input/tips.csv', encoding='latin1')

df.head()
df['y'] = pd.factorize(df.Result)[0]

df['tipster'] = pd.factorize(df.Tipster)[0]

df['track'] = pd.factorize(df.Track)[0]

df['bet_type'] = pd.factorize(df['Bet Type'])[0]

df['year'] = pd.DatetimeIndex(df['Date']).year
X = df[['Odds', 'tipster','bet_type', 'year', 'track', 'y']]

X, test = train_test_split(X, test_size = 0.2)

y = X.pop('y')

test_y = test.pop('y')

X.head()
# needs a bit more tuning

cv_params = {'max_depth': [3,5], 'min_child_weight': [1, 3], 

             'n_estimators': [100], 'learning_rate': [.1]}

optimized_GBM = GridSearchCV(xgb.XGBClassifier(), 

                            cv_params, 

                             scoring = 'accuracy', cv = 5, n_jobs = -1) 

optimized_GBM.fit(X,y)
optimized_GBM.grid_scores_
optimized_GBM.best_params_
xgb.plot_importance(optimized_GBM.best_estimator_)