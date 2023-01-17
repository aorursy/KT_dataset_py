import numpy as np

import pandas as pd



from sklearn.metrics.regression import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.linear_model import LinearRegression, LassoCV, Lasso

from sklearn.ensemble import RandomForestRegressor
wines = pd.read_csv('../input/winequality-white.csv')

print(wines.shape)

wines.head()
wines.info()
X = wines.drop('quality', axis='columns')

y = wines['quality']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

X_train_scaled.shape, X_test_scaled.shape
linreg = LinearRegression()

linreg.fit(X_train_scaled, y_train)
y_train_pred = linreg.predict(X_train_scaled)

y_test_pred = linreg.predict(X_test_scaled)

print('Mean squared error (train): {:.3f}'.format(mean_squared_error(y_train, y_train_pred)))

print('Mean squared error (test): {:.3f}'.format(mean_squared_error(y_test, y_test_pred)))
importances = pd.DataFrame(abs(linreg.coef_),

                           index=X.columns,

                           columns=['importance'])

importances.sort_values('importance').plot(kind='barh')
lasso = Lasso(alpha=0.01, random_state=17)

lasso.fit(X_train_scaled, y_train)
importances = pd.DataFrame(abs(lasso.coef_),

                           index=X.columns,

                           columns=['importance'])

importances.sort_values('importance').plot(kind='barh')
alphas = np.logspace(-6, 2, 200)

print('[', min(alphas), max(alphas), ']')



lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=17, n_jobs=-1)

lasso_cv.fit(X_train_scaled, y_train)

lasso_cv.alpha_    # alpha chosen by cross validation
importances = pd.DataFrame(abs(lasso_cv.coef_),

                           index=X.columns,

                           columns=['importance'])

importances.sort_values('importance').plot(kind='barh')
y_train_pred = lasso_cv.predict(X_train_scaled)

y_test_pred = lasso_cv.predict(X_test_scaled)

print('Mean squared error (train): {:.3f}'.format(mean_squared_error(y_train, y_train_pred)))

print('Mean squared error (test): {:.3f}'.format(mean_squared_error(y_test, y_test_pred)))
forest = RandomForestRegressor(n_estimators=10, random_state=17)

forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)

cv_scores = cross_val_score(forest, X_train, y_train,

                            scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

y_test_pred = forest.predict(X_test)

print('Mean squared error (train): {:.3f}'.format(mean_squared_error(y_train, y_train_pred)))

print('Mean squared error (cv): {:.3f}'.format(abs(cv_scores).mean()))

print('Mean squared error (test): {:.3f}'.format(mean_squared_error(y_test, y_test_pred)))
%%time

forest_params = {'max_depth': range(10, 25), 

                 'min_samples_leaf': range(1, 8),

                 'max_features': range(6,12)}



locally_best_forest = GridSearchCV(forest, forest_params,

                                   scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

locally_best_forest.fit(X_train, y_train)

print(abs(locally_best_forest.best_score_), locally_best_forest.best_params_)
cv_scores = cross_val_score(locally_best_forest.best_estimator_, X_train, y_train,

                            scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

y_test_pred = locally_best_forest.predict(X_test)

print('Mean squared error (cv): {:.3f}'.format(abs(cv_scores).mean()))

print('Mean squared error (test): {:.3f}'.format(mean_squared_error(y_test, y_test_pred)))
importances = pd.DataFrame(locally_best_forest.best_estimator_.feature_importances_,

                           index=X.columns,

                           columns=['importance'])

importances.sort_values('importance').plot(kind='barh')