import pandas as pd

import sklearn as sk

import numpy as np

import scipy as sp

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import explained_variance_score, mean_squared_error
df = pd.read_csv('../input/vgsales.csv')
df = df.sort_values(['Global_Sales'], ascending=False)

df = df[~df[['Name', 'Platform', 'Year']].duplicated(keep='first')]
df = df[~df.isnull()['Year']]
len(df)
df_features = pd.get_dummies(df.Platform)

df_features = df_features.join(pd.get_dummies(df.Genre), how='outer', lsuffix='_left', rsuffix='_right')

df_features = df_features.join(pd.get_dummies(df.Publisher), how='outer', lsuffix='_left', rsuffix='_right')

df_features['Year'] = df['Year']

len(df_features)
X = np.array(df_features)
y = np.array(df.Global_Sales)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr1 = LinearRegression()
lr1_scores = cross_val_score(lr1, X, y, 

                             scoring='neg_mean_squared_error',

                             cv=10,

                             n_jobs=-1,

                             verbose=1)
lr1_scores
rf1 = RandomForestRegressor(n_estimators=10)
rf1_scores = cross_val_score(rf1, X, y, 

                             scoring='neg_mean_squared_error',

                             cv=10,

                             n_jobs=-1,

                             verbose=1)
rf1_scores
rf_params = {

    'n_estimators': [50, 200, 400],

    'min_samples_leaf': [2, 5, 10],

    'max_features': ['auto', 'sqrt']

}



rf_gs = GridSearchCV(RandomForestRegressor(),

                     param_grid=rf_params,

                     scoring='neg_mean_squared_error',

                     n_jobs=-1,

                     verbose=2,

                     cv = 10)
rf_gs.fit(X, y)
rf_gs.best_params_
rf2 = rf_gs.best_estimator_
rf2_scores = cross_val_score(rf2, X, y, 

                             scoring='neg_mean_squared_error',

                             cv=10,

                             n_jobs=-1,

                             verbose=1) 
rf2_scores
model_names = ['simple LR', 

               'simple RF',

               'tuned RF']

model_scores = [lr1_scores,

                rf1_scores,

                rf2_scores]

results = pd.DataFrame(model_scores, index=model_names).transpose()



results.describe()
results.boxplot()

plt.show()
results.boxplot()

axes = plt.gca()

axes.set_ylim([-30,0])

plt.show()
results.boxplot()

axes = plt.gca()

axes.set_ylim([-0.5,0])

plt.show()
np.percentile(lr1_scores, 50)
df_features.Year.unique()