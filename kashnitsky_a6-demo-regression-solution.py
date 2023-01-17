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
X = data.drop('quality', axis=1)

X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3, 
                                                          random_state=17)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_holdout_scaled = scaler.transform(X_holdout)
linreg = LinearRegression()
linreg.fit(X_train_scaled, y_train);
print("Mean squared error (train): %.3f" % mean_squared_error(y_train, linreg.predict(X_train_scaled)))
print("Mean squared error (test): %.3f" % mean_squared_error(y_holdout, linreg.predict(X_holdout_scaled)))
linreg_coef = pd.DataFrame({'coef': linreg.coef_, 'coef_abs': np.abs(linreg.coef_)},
                          index=data.columns.drop('quality'))
linreg_coef.sort_values(by='coef_abs', ascending=False)
lasso1 = Lasso(alpha=0.01, random_state=17)
lasso1.fit(X_train_scaled, y_train)
lasso1_coef = pd.DataFrame({'coef': lasso1.coef_, 'coef_abs': np.abs(lasso1.coef_)},
                          index=data.columns.drop('quality'))
lasso1_coef.sort_values(by='coef_abs', ascending=False)
alphas = np.logspace(-6, 2, 200)
lasso_cv = LassoCV(random_state=17, cv=5, alphas=alphas)
lasso_cv.fit(X_train_scaled, y_train)
lasso_cv.alpha_
lasso_cv_coef = pd.DataFrame({'coef': lasso_cv.coef_, 'coef_abs': np.abs(lasso_cv.coef_)},
                          index=data.columns.drop('quality'))
lasso_cv_coef.sort_values(by='coef_abs', ascending=False)
print("Mean squared error (train): %.3f" % mean_squared_error(y_train, lasso_cv.predict(X_train_scaled)))
print("Mean squared error (test): %.3f" % mean_squared_error(y_holdout, lasso_cv.predict(X_holdout_scaled)))
forest = RandomForestRegressor(random_state=17)
forest.fit(X_train_scaled, y_train)
print("Mean squared error (train): %.3f" % mean_squared_error(y_train, forest.predict(X_train_scaled)))
print("Mean squared error (cv): %.3f" % np.mean(np.abs(cross_val_score(forest, X_train_scaled, y_train, 
                                                                       scoring='neg_mean_squared_error'))))
print("Mean squared error (test): %.3f" % mean_squared_error(y_holdout, forest.predict(X_holdout_scaled)))


forest_params = {'max_depth': list(range(10, 25)), 
                  'max_features': list(range(6,12))}

locally_best_forest = GridSearchCV(RandomForestRegressor(n_jobs=-1, random_state=17), 
                                 forest_params, 
                                 scoring='neg_mean_squared_error',  
                                 n_jobs=-1, cv=5,
                                  verbose=True)
locally_best_forest.fit(X_train_scaled, y_train)
locally_best_forest.best_params_, locally_best_forest.best_score_
print("Mean squared error (cv): %.3f" % np.mean(np.abs(cross_val_score(locally_best_forest.best_estimator_,
                                                        X_train_scaled, y_train, 
                                                        scoring='neg_mean_squared_error'))))
print("Mean squared error (test): %.3f" % mean_squared_error(y_holdout, 
                                                             locally_best_forest.predict(X_holdout_scaled)))
rf_importance = pd.DataFrame(locally_best_forest.best_estimator_.feature_importances_, 
                             columns=['coef'], index=data.columns[:-1]) 
rf_importance.sort_values(by='coef', ascending=False)