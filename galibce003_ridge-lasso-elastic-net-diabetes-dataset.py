import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings

warnings.filterwarnings('ignore', category = DeprecationWarning)
from warnings import filterwarnings
filterwarnings('ignore')
from sklearn.datasets import load_diabetes
df = load_diabetes()
df.keys()
unscaled_X = df.data
y = df.target
scaler = StandardScaler()
X = scaler.fit_transform(unscaled_X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
lr = LinearRegression()

# Train the model
model_lr = lr.fit(X_train, y_train)

# Prediction
y_pred_train_lr = lr.predict(X_train)
y_pred_test_lr = lr.predict(X_test)

# Accuracy Score
print('Training accuracy : {}\n'.format(r2_score(y_train, y_pred_train_lr).round(5)))
print('Testing accuracy : {}\n'.format(r2_score(y_test, y_pred_test_lr).round(5)))
mse=cross_val_score(lr, X, y, scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse)
print(-(mean_mse).round(5))
coefs = pd.Series(lr.coef_, index = df.feature_names)
coefs.plot(kind = 'barh', cmap = 'autumn')
plt.show()
ridge1 = Ridge()

# Fit the model
model_ridge1 = ridge1.fit(X_train, y_train)

# Prediction
y_pred_train_ridge1 = ridge1.predict(X_train)
y_pred_test_ridge1 = ridge1.predict(X_test)

# Accuracy Score
print('Training accuracy : {}\n'.format(r2_score(y_train, y_pred_train_ridge1).round(5)))
print('Testing accuracy : {}'.format(r2_score(y_test, y_pred_test_ridge1).round(5)))
mse1 = cross_val_score(ridge1, X, y, scoring = 'neg_mean_squared_error',cv=5)
mean_mse1 = np.mean(mse1)
print(-(mean_mse1).round(5))
coefs_ridge1 = pd.Series(ridge1.coef_, index = df.feature_names)
coefs_ridge1.plot(kind = 'barh', cmap = 'gist_heat')
plt.show()
ridge2 = Ridge()

parameters = {'alpha' : [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 0.9, 0.8, 0.7, 0.5, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}

ridge_regressor = GridSearchCV(ridge2, parameters, scoring='neg_mean_squared_error', cv= 10)

ridge_regressor.fit(X_train, y_train)
print(ridge_regressor.best_params_)
print(-(ridge_regressor.best_score_))
# Prediction
y_train_pred_ridg2 = ridge_regressor.predict(X_train)
y_test_pred_ridg2 = ridge_regressor.predict(X_test)

# Accuracy Score
print('Training accuracy : {}\n'.format(r2_score(y_train, y_train_pred_ridg2).round(5)))
print('Testing accuracy : {}'.format(r2_score(y_test, y_test_pred_ridg2).round(5)))
mse2 = cross_val_score(ridge_regressor, X, y, scoring = 'neg_mean_squared_error',cv=5)
mean_mse2 = np.mean(mse2)
print(-(mean_mse2).round(5))
lasso1 = Lasso()

# Fit the model
model_lasso1 = lasso1.fit(X_train, y_train)

# Prediction
y_pred_train_lasso1 = lasso1.predict(X_train)
y_pred_test_lasso1 = lasso1.predict(X_test)

# Accuracy Score
print('Training accuracy : {}\n'.format(r2_score(y_train, y_pred_train_lasso1).round(5)))
print('Testing accuracy : {}'.format(r2_score(y_test, y_pred_test_lasso1).round(5)))
mse3 = cross_val_score(lasso1, X, y, scoring = 'neg_mean_squared_error',cv=5)
mean_mse3 = np.mean(mse3)
print(-(mean_mse3).round(5))
coefs_lasso1 = pd.Series(lasso1.coef_, index = df.feature_names)
coefs_lasso1.plot(kind = 'barh', cmap = 'autumn')
plt.show()
lasso2 = Lasso()

parameters = {'alpha' : [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 0.9, 0.8, 0.7, 0.5, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}

lasso_regressor = GridSearchCV(lasso2, parameters, scoring='neg_mean_squared_error', cv= 10)

lasso_regressor.fit(X_train, y_train)
print(lasso_regressor.best_params_)
print(-(lasso_regressor.best_score_))
mse4 = cross_val_score(lasso_regressor, X, y, scoring = 'neg_mean_squared_error',cv=5)
mean_mse4 = np.mean(mse4)
print(-(mean_mse4))
# Prediction
y_train_pred_lasso2 = lasso_regressor.predict(X_train)
y_test_pred_lasso2 = lasso_regressor.predict(X_test)

# Accuracy Score
print('Training accuracy : {}\n'.format(r2_score(y_train, y_train_pred_lasso2).round(5)))
print('Testing accuracy : {}'.format(r2_score(y_test, y_test_pred_lasso2).round(5)))
enet1 = ElasticNet()

# Fit the model
model_enet1 = enet1.fit(X_train, y_train)

# Prediction
y_pred_train_enet1 = enet1.predict(X_train)
y_pred_test_enet1 = enet1.predict(X_test)

# Accuracy Score
print('Training accuracy : {}\n'.format(r2_score(y_train, y_pred_train_enet1).round(5)))
print('Testing accuracy : {}'.format(r2_score(y_test, y_pred_test_enet1).round(5)))
mse5 = cross_val_score(enet1, X, y, scoring = 'neg_mean_squared_error',cv=5)
mean_mse5 = np.mean(mse5)
print(-(mean_mse5).round(5))
coefs_enet1 = pd.Series(enet1.coef_, index = df.feature_names)
coefs_enet1.plot(kind = 'barh', cmap = 'autumn')
plt.show()
enet2 = ElasticNet()

parameters = {'alpha' : [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 0.9, 0.8, 0.7, 0.5, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}

enet_regressor = GridSearchCV(enet2, parameters, scoring='neg_mean_squared_error', cv= 10)

enet_regressor.fit(X_train, y_train)
print(enet_regressor.best_params_)
print(-(enet_regressor.best_score_))
mse5 = cross_val_score(enet_regressor, X, y, scoring = 'neg_mean_squared_error',cv=5)
mean_mse5 = np.mean(mse5)
print(-(mean_mse5))
# Prediction
y_train_pred_enet2 = enet_regressor.predict(X_train)
y_test_pred_enet2 = enet_regressor.predict(X_test)

# Accuracy Score
print('Training accuracy : {}\n'.format(r2_score(y_train, y_train_pred_enet2).round(5)))
print('Testing accuracy : {}'.format(r2_score(y_test, y_test_pred_enet2).round(5)))