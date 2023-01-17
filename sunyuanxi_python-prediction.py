import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
%matplotlib inline
# Read Dataset

df = pd.read_csv('../input/Car_sales.csv')
df = df.dropna()
print(df.shape)
df.head()
# Change 'Passenger' to binary indicators

df['Passenger'] = (df['Vehicle_type']=='Passenger')
df['Car'] = (df['Vehicle_type']!='Passenger')
df.drop('Vehicle_type', inplace=True, axis=1)
df.head()
df.corr()
X_train, X_test, y_train, y_test = train_test_split(df.drop('Sales_in_thousands', axis=1), df['Sales_in_thousands'], random_state = 0)
linreg = LinearRegression().fit(X_train, y_train)

print('linear model coeff (w): {}'.format(linreg.coef_))
print('linear model intercept (b): {:.3f}'.format(linreg.intercept_))
print('R-squared score (training): {:.3f}'.format(linreg.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(linreg.score(X_test, y_test)))
knn = KNeighborsRegressor().fit(X_train, y_train)
print('R-squared score (training): {:.3f}'.format(knn.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(knn.score(X_test, y_test)))
grid_values = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
grid_knn_mse = GridSearchCV(knn, param_grid = grid_values, scoring = 'neg_mean_squared_error', cv=5, iid=False)
grid_knn_mse.fit(X_train, y_train)

print('Grid best parameter (min. MSE): ', grid_knn_mse.best_params_)
print('Grid best score (MSE): ', grid_knn_mse.best_score_)
knn = grid_knn_mse.best_estimator_
print('R-squared score (training): {:.3f}'.format(knn.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(knn.score(X_test, y_test)))
svm = SVR(gamma='scale').fit(X_train, y_train)
print('R-squared score (training): {:.3f}'.format(svm.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(svm.score(X_test, y_test)))
grid_values = {'gamma': [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],\
               'C': [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]}
grid_svm_mse = GridSearchCV(svm, param_grid = grid_values, scoring = 'neg_mean_squared_error', cv=5, iid=False)
grid_svm_mse.fit(X_train, y_train)

print('Grid best parameter (min. MSE): ', grid_svm_mse.best_params_)
print('Grid best score (MSE): ', grid_svm_mse.best_score_)
svm = grid_svm_mse.best_estimator_
print('R-squared score (training): {:.3f}'.format(svm.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(svm.score(X_test, y_test)))
dt = DecisionTreeRegressor().fit(X_train, y_train)
print('R-squared score (training): {:.3f}'.format(dt.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(dt.score(X_test, y_test)))
plt.figure(figsize=(10,10), dpi=80)
feature_names = X_train.columns
feature_importance = pd.DataFrame(feature_names, columns=['features'])
feature_importance['importance'] = pd.DataFrame(dt.feature_importances_)
feature_importance.sort_values(by='importance', ascending=False, inplace=True)
feature_importance.reset_index(drop=True, inplace=True)
plt.barh(feature_importance['features'], feature_importance['importance'])
plt.xlabel('Feature importance')
plt.ylabel('Feature name')
plt.show()
grid_values = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],\
               'min_samples_split': [2, 4, 8, 16, 32, 64, 100],\
               'min_samples_leaf': [1, 2, 4, 8, 16, 32, 64, 100]}
grid_dt_mse = GridSearchCV(dt, param_grid = grid_values, scoring = 'neg_mean_squared_error', cv=5, iid=False)
grid_dt_mse.fit(X_train, y_train)

print('Grid best parameter (min. MSE): ', grid_dt_mse.best_params_)
print('Grid best score (MSE): ', grid_dt_mse.best_score_)
dt = grid_dt_mse.best_estimator_
print('R-squared score (training): {:.3f}'.format(dt.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(dt.score(X_test, y_test)))
plt.figure(figsize=(10,10), dpi=80)
feature_names = X_train.columns
feature_importance = pd.DataFrame(feature_names, columns=['features'])
feature_importance['importance'] = pd.DataFrame(dt.feature_importances_)
feature_importance.sort_values(by='importance', ascending=False, inplace=True)
feature_importance.reset_index(drop=True, inplace=True)
plt.barh(feature_importance['features'], feature_importance['importance'])
plt.xlabel('Feature importance')
plt.ylabel('Feature name')
plt.show()
rf = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
print('R-squared score (training): {:.3f}'.format(rf.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(rf.score(X_test, y_test)))
grid_values = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],\
               'min_samples_split': [2, 4, 8, 16, 32, 64, 100],\
               'min_samples_leaf': [1, 2, 4, 8, 16, 32, 64, 100]}
grid_rf_mse = GridSearchCV(rf, param_grid = grid_values, scoring = 'neg_mean_squared_error', cv=5, iid=False)
grid_rf_mse.fit(X_train, y_train)

print('Grid best parameter (min. MSE): ', grid_rf_mse.best_params_)
print('Grid best score (MSE): ', grid_rf_mse.best_score_)
rf = grid_rf_mse.best_estimator_
print('R-squared score (training): {:.3f}'.format(rf.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(rf.score(X_test, y_test)))
plt.figure(figsize=(10,10), dpi=80)
feature_names = X_train.columns
feature_importance = pd.DataFrame(feature_names, columns=['features'])
feature_importance['importance'] = pd.DataFrame(rf.feature_importances_)
feature_importance.sort_values(by='importance', ascending=False, inplace=True)
feature_importance.reset_index(drop=True, inplace=True)
plt.barh(feature_importance['features'], feature_importance['importance'])
plt.xlabel('Feature importance')
plt.ylabel('Feature name')
plt.show()
nn = MLPRegressor().fit(X_train, y_train)
print('R-squared score (training): {:.3f}'.format(nn.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(nn.score(X_test, y_test)))
grid_values = {'hidden_layer_sizes': np.arange(1, 201),\
               'learning_rate': ['constant', 'invscaling', 'adaptive']}
grid_nn_mse = GridSearchCV(nn, param_grid = grid_values, scoring = 'neg_mean_squared_error', cv=5, iid=False)
grid_nn_mse.fit(X_train, y_train)

print('Grid best parameter (min. MSE): ', grid_nn_mse.best_params_)
print('Grid best score (MSE): ', grid_nn_mse.best_score_)
nn = grid_nn_mse.best_estimator_
print('R-squared score (training): {:.3f}'.format(nn.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(nn.score(X_test, y_test)))
xg_reg = xgb.XGBRegressor().fit(X_train, y_train)
print('R-squared score (training): {:.3f}'.format(xg_reg.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(xg_reg.score(X_test, y_test)))
grid_values = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],\
               'min_child_weight': [1, 2, 4, 8, 16, 32, 64, 100]}
grid_xgb_mse = GridSearchCV(xg_reg, param_grid = grid_values, scoring = 'neg_mean_squared_error', cv=5, iid=False)
grid_xgb_mse.fit(X_train, y_train)

print('Grid best parameter (min. MSE): ', grid_xgb_mse.best_params_)
print('Grid best score (MSE): ', grid_xgb_mse.best_score_)
xg_reg = grid_xgb_mse.best_estimator_
print('R-squared score (training): {:.3f}'.format(xg_reg.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(xg_reg.score(X_test, y_test)))
gb = GradientBoostingRegressor().fit(X_train, y_train)
print('R-squared score (training): {:.3f}'.format(gb.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(gb.score(X_test, y_test)))
grid_values = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],\
               'min_samples_split': [2, 4, 8, 16, 32, 64, 100],\
               'min_samples_leaf': [1, 2, 4, 8, 16, 32, 64, 100]}
grid_gb_mse = GridSearchCV(gb, param_grid = grid_values, scoring = 'neg_mean_squared_error', cv=5, iid=False)
grid_gb_mse.fit(X_train, y_train)

print('Grid best parameter (min. MSE): ', grid_gb_mse.best_params_)
print('Grid best score (MSE): ', grid_gb_mse.best_score_)
gb = grid_gb_mse.best_estimator_
print('R-squared score (training): {:.3f}'.format(gb.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(gb.score(X_test, y_test)))
plt.figure(figsize=(10,10), dpi=80)
feature_names = X_train.columns
feature_importance = pd.DataFrame(feature_names, columns=['features'])
feature_importance['importance'] = pd.DataFrame(gb.feature_importances_)
feature_importance.sort_values(by='importance', ascending=False, inplace=True)
feature_importance.reset_index(drop=True, inplace=True)
plt.barh(feature_importance['features'], feature_importance['importance'])
plt.xlabel('Feature importance')
plt.ylabel('Feature name')
plt.show()
stacking = pd.DataFrame(y_train)
models = [linreg, knn, svm, dt, nn, rf, gb, xg_reg]
for i in models:
    stacking[str(i)] = i.predict(X_train)
stacking.columns = ['Sales_in_thousands', 'LinReg', 'kNN', 'SVM', 'Decision Tree', 'Neural Network', 'Random Forest', 'Gradient Boosting', 'XGB']
stacking.sort_values('Sales_in_thousands', inplace=True)
linreg_s = LinearRegression().fit(stacking.drop('Sales_in_thousands', axis=1), stacking['Sales_in_thousands'])
coef = linreg_s.coef_
print(coef)
prediction = df.iloc[:, :1]
models = [linreg, knn, svm, dt, nn, rf, gb, xg_reg]
for i in models:
    prediction[str(i)] = i.predict(df.drop('Sales_in_thousands', axis=1))
prediction.columns = ['Sales_in_thousands', 'LinReg', 'kNN', 'SVM', 'Decision Tree', 'Neural Network', 'Random Forest', 'Gradient Boosting', 'XGB']
prediction['Stacking'] = coef[0]*prediction['LinReg'] + coef[1]*prediction['kNN'] + coef[2]*prediction['SVM']\
                       + coef[3]*prediction['Decision Tree'] + coef[4]*prediction['Neural Network']\
                       + coef[5]*prediction['Random Forest'] + coef[6]*prediction['Gradient Boosting']\
                       + coef[7]*prediction['XGB']
prediction.sort_values('Sales_in_thousands', inplace=True)
prediction.head()
score = []
for i in range(1, 10):
    score += [((prediction.iloc[:, i] - prediction['Sales_in_thousands'])**2).mean()]
best_model = score.index(min(score))
print(score)
fig = plt.figure(figsize=(10, 6))
plt.plot(np.arange(prediction.shape[0]), prediction['Sales_in_thousands'], label='True Value', linewidth=3)
for i in range(len(prediction.columns[1:])):
    if i == best_model:
        plt.plot(np.arange(prediction.shape[0]), prediction.iloc[:, i+1], label=prediction.columns[i+1]+'('+str(round(score[i], 2))+')', linewidth=3)
    else:
        plt.plot(np.arange(prediction.shape[0]), prediction.iloc[:, i+1], label=prediction.columns[i+1]+'('+str(round(score[i], 2))+')')
plt.title('Prediction Results', fontsize=16)
plt.xticks(fontsize=13)
plt.xlabel('Instances', fontsize=16)
plt.yticks(fontsize=13)
plt.ylabel('Sales in thousands', fontsize=16)
plt.xlim(0, prediction.shape[0]-1)
plt.grid()
plt.legend(loc=2)
plt.show()