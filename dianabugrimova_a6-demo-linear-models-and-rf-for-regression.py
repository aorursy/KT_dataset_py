import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

from sklearn.metrics.regression import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.linear_model import LinearRegression, LassoCV, Lasso

from sklearn.ensemble import RandomForestRegressor
data = pd.read_csv('../input/winequality-white.csv')
data.head()
data.info()
y = data['quality']

X = data.drop(['quality'], axis=1)

X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.30, random_state=17)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_holdout_scaled = scaler.transform(X_holdout)
linreg = LinearRegression()

linreg.fit(X_train_scaled,y_train)
y_train_pred=linreg.predict(X_train_scaled)
y_pred=linreg.predict(X_holdout_scaled)
mean_squared_error(y_holdout, y_pred)
print("Mean squared error (train):", mean_squared_error(y_train, y_train_pred))

print("Mean squared error (test):" ,mean_squared_error(y_holdout, y_pred))
X.columns
linreg_coef = pd.DataFrame({'feat': X.columns,

              'coef': linreg.coef_.flatten().tolist()}).sort_values(by='coef', ascending=False)

linreg_coef
lasso1 = Lasso(alpha=0.01, random_state=17)

lasso1.fit(X_train_scaled,y_train)
lasso1_coef = pd.DataFrame({'feat': X.columns,

              'coef': lasso1.coef_.flatten().tolist()}).sort_values(by='coef', ascending=False)

lasso1_coef
alphas = np.logspace(-6, 2, 200)

lasso_cv = LassoCV(random_state=17, alphas=alphas)

lasso_cv.fit(X_train_scaled,y_train)
lasso_cv.alpha_
lasso_cv_coef = pd.DataFrame({'feat': X.columns,

              'coef': lasso_cv.coef_.flatten().tolist()}).sort_values(by='coef', ascending=False)

lasso_cv_coef
y_pred=lasso_cv.predict(X_holdout_scaled)
y_train_pred=lasso_cv.predict(X_train_scaled)
print("Mean squared error (train):", mean_squared_error(y_train, y_train_pred))

print("Mean squared error (test):" ,mean_squared_error(y_holdout, y_pred))
forest = RandomForestRegressor(random_state=17)

forest.fit(X_train_scaled,y_train)
y_pred=forest.predict(X_holdout_scaled)
y_train_pred=lasso_cv.predict(X_train_scaled)
cross_val_score(forest, X_train_scaled, y_train, scoring='neg_mean_squared_error').mean()
print("Mean squared error (train):", mean_squared_error(y_train, y_train_pred))

print("Mean squared error (cv):",cross_val_score(forest, X_train_scaled, y_train, scoring='neg_mean_squared_error').mean())

print("Mean squared error (test):", mean_squared_error(y_holdout, y_pred))
forest_params = {'max_depth': list(range(10, 25)), 

                 'min_samples_leaf': list(range(1, 8)),

                  'max_features': list(range(6,12))}



locally_best_forest = GridSearchCV(forest, forest_params)

locally_best_forest.fit(X_train_scaled,y_train)
locally_best_forest.best_params_, locally_best_forest.best_score_
y_pred=locally_best_forest.predict(X_holdout_scaled)
print("Mean squared error (cv):", cross_val_score(locally_best_forest, X_train_scaled, y_train, scoring='neg_mean_squared_error').mean())

print("Mean squared error (test):",mean_squared_error(y_holdout, y_pred))
rf_importance = pd.DataFrame ({'feat': X.columns,

              'coef': lasso_cv.coef_.flatten().tolist()}).sort_values(by='coef', ascending=False)

rf_importance