import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_predict
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
df.replace('.', np.nan, inplace=True)
df = df.dropna()
print(df.shape)
df.head()
# Change 'Passenger' to binary indicators
# Convert object to float64

df['Passenger'] = (df['Vehicle type']=='Passenger')
df.drop('Vehicle type', inplace=True, axis=1)

for i in df.columns[2:-2]:
    df[i] = pd.to_numeric(df[i])
# Convert 'Latest Launch' to the number of days after 1-Jan-1970

df['Latest Launch'] = pd.to_datetime(df['Latest Launch'])
df['Latest Launch'] = (df['Latest Launch'] - pd.Timestamp(1970, 1, 1)).astype(str)
for i in df.index:
    df.loc[i, 'Latest Launch'] = float(df.loc[i, 'Latest Launch'].split()[0])
# Convert 'Manufacturer' as binary indicators.

for i in df['Manufacturer'].unique()[:-1]:
    df[i] = (df['Manufacturer'] == i)
df.head(10)
fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)
ax.set_axisbelow(True)
plt.hist(df['Sales in thousands'], bins=25, rwidth=0.8)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.tick_params(left=False)
ax.set_xlabel('Sales in thousands', fontsize=16)
ax.set_ylabel('Count', fontsize=16)
ax.set_title('Distribution of Target Variable', fontsize=16)
plt.grid(axis='y')
plt.show()
df = df[df['Sales in thousands'] <= 300]
df.drop(['4-year resale value', 'Latest Launch'], axis=1, inplace=True)
df.shape
df.head(10)
round(df.iloc[:, 2:].corr(), 4)
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Manufacturer', 'Model', 'Sales in thousands'], axis=1), df['Sales in thousands'], random_state = 0)
linreg = LinearRegression().fit(X_train, y_train)

print('linear model coeff (w): {}'.format(linreg.coef_))
print('linear model intercept (b): {:.3f}'.format(linreg.intercept_))
print('R-squared score (training): {:.3f}'.format(linreg.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(linreg.score(X_test, y_test)))
knn = KNeighborsRegressor().fit(X_train, y_train)
print('R-squared score (training): {:.3f}'.format(knn.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(knn.score(X_test, y_test)))
grid_values = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
kf = KFold(n_splits=5, random_state=13)
grid_knn_mse = GridSearchCV(knn, param_grid = grid_values, scoring = 'neg_mean_squared_error', cv=kf, iid=False)
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
               'C': [30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]}
grid_svm_mse = GridSearchCV(svm, param_grid = grid_values, scoring = 'neg_mean_squared_error', cv=kf, iid=False)
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
grid_dt_mse = GridSearchCV(dt, param_grid = grid_values, scoring = 'neg_mean_squared_error', cv=kf, iid=False)
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
rf = RandomForestRegressor(n_estimators=100, random_state=3).fit(X_train, y_train)
print('R-squared score (training): {:.3f}'.format(rf.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(rf.score(X_test, y_test)))
grid_values = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],\
               'min_samples_split': [2, 4, 8, 16, 32, 64, 100],\
               'min_samples_leaf': [1, 2, 4, 8, 16, 32, 64, 100]}
grid_rf_mse = GridSearchCV(rf, param_grid = grid_values, scoring = 'neg_mean_squared_error', cv=kf, iid=False, return_train_score=False)
grid_rf_mse.fit(X_train, y_train)

print('Grid best parameter (min. MSE): ', grid_rf_mse.best_params_)
print('Grid best score (MSE): ', grid_rf_mse.best_score_)
cv_results_rf = pd.DataFrame(grid_rf_mse.cv_results_)
cv_results_rf = cv_results_rf[cv_results_rf['param_max_depth']==grid_rf_mse.best_params_['max_depth']]
rf_score = np.array(cv_results_rf['mean_test_score']).reshape(8, 7)
fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)
im = ax.imshow(rf_score, cmap='cividis')
cbar = ax.figure.colorbar(im, ax=ax)
ax.set_xticks(np.arange(7))
ax.set_yticks(np.arange(8))
ax.set_xticklabels(grid_values['min_samples_split'], fontsize=13)
ax.set_yticklabels(grid_values['min_samples_leaf'], fontsize=13)
for i in range(7):
    for j in range(8):
        text = ax.text(i, j, round(rf_score[j][i]), ha="center", va="center", color="w")
ax.set_xlabel('min_samples_split', fontsize=16)
ax.set_ylabel('min_samples_leaf', fontsize=16)
ax.set_title('Random Forest Grid Search', fontsize=16)
plt.show()
rf = grid_rf_mse.best_estimator_
print('R-squared score (training): {:.3f}'.format(rf.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(rf.score(X_test, y_test)))
plt.figure(figsize=(10,6), dpi=80)
ax = plt.subplot(111)
feature_names = X_train.columns
feature_importance = pd.DataFrame(feature_names, columns=['features'])
feature_importance['importance'] = pd.DataFrame(rf.feature_importances_)
feature_importance.sort_values(by='importance', ascending=False, inplace=True)
feature_importance.reset_index(drop=True, inplace=True)
feature_importance['features'].replace(['Ford         ', 'Honda        ', 'Toyota       '], ['Ford', 'Honda', 'Toyota'], inplace=True)
bars = plt.barh(feature_importance['features'][:10], feature_importance['importance'][:10])
plt.tick_params(left=False, bottom=False, labelbottom=False)
for spine in plt.gca().spines.values():
    spine.set_visible(False)
for bar in bars:
    plt.gca().text(bar.get_width() - 0.01, bar.get_y() + bar.get_height()/3, str(round(bar.get_width(),3)), 
                   ha='center', color='white', fontsize=11)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('Random Forest Feature importance (Top 10)', fontsize=16)
plt.ylabel('Feature name', fontsize=16)
plt.show()
#nn = MLPRegressor().fit(X_train, y_train)
#print('R-squared score (training): {:.3f}'.format(nn.score(X_train, y_train)))
#print('R-squared score (test): {:.3f}'.format(nn.score(X_test, y_test)))
#grid_values = {'hidden_layer_sizes': np.arange(1, 201),\
#               'learning_rate': ['constant', 'invscaling', 'adaptive']}
#grid_nn_mse = GridSearchCV(nn, param_grid = grid_values, scoring = 'neg_mean_squared_error', cv=5, iid=False)
#grid_nn_mse.fit(X_train, y_train)

#print('Grid best parameter (min. MSE): ', grid_nn_mse.best_params_)
#print('Grid best score (MSE): ', grid_nn_mse.best_score_)
#nn = MLPRegressor(hidden_layer_sizes=200, learning_rate='invscaling').fit(X_train, y_train)
#print('R-squared score (training): {:.3f}'.format(nn.score(X_train, y_train)))
#print('R-squared score (test): {:.3f}'.format(nn.score(X_test, y_test)))
#nn = grid_nn_mse.best_estimator_
#print('R-squared score (training): {:.3f}'.format(nn.score(X_train, y_train)))
#print('R-squared score (test): {:.3f}'.format(nn.score(X_test, y_test)))
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
gb = GradientBoostingRegressor(random_state=3).fit(X_train, y_train)
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
models = [linreg, knn, svm, dt, rf, gb, xg_reg]
for i in models:
    stacking[str(i)] = i.predict(X_train)
stacking.columns = ['Sales in thousands', 'LinReg', 'kNN', 'SVM', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGB']
stacking.sort_values('Sales in thousands', inplace=True)
linreg_s = LinearRegression().fit(stacking.drop('Sales in thousands', axis=1), stacking['Sales in thousands'])
coef = linreg_s.coef_
print(coef)
prediction = pd.DataFrame(y_test)
models = [linreg, knn, svm, dt, rf, gb, xg_reg]
for i in models:
    prediction[str(i)] = i.predict(X_test)
prediction.columns = ['Sales in thousands', 'LinReg', 'kNN', 'SVM', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGB']
prediction['Stacking'] = coef[0]*prediction['LinReg'] + coef[1]*prediction['kNN'] + coef[2]*prediction['SVM']\
                       + coef[3]*prediction['Decision Tree'] + coef[4]*prediction['Random Forest']\
                       + coef[5]*prediction['Gradient Boosting'] + coef[6]*prediction['XGB']\
                       + linreg_s.intercept_
#prediction.sort_values('Sales in thousands', inplace=True)
prediction.head()
score = []
for i in range(1, 9):
    score += [((prediction.iloc[:, i] - prediction['Sales in thousands'])**2).mean()]
best_model = score.index(min(score))
pd.Series(score, index = ['LinReg', 'kNN', 'SVM', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGB', 'Stacking'])
fig = plt.figure(figsize=(10, 6))
plt.plot(np.arange(prediction.shape[0]), prediction['Sales in thousands'], label='True Value', linewidth=3)
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
prediction = df.iloc[:, 2:3]
models = [linreg, knn, svm, dt, rf, gb, xg_reg]
for i in models:
    prediction[str(i)] = cross_val_predict(i, df.drop(['Manufacturer', 'Model', 'Sales in thousands'], axis=1), df['Sales in thousands'], cv=5)
prediction.columns = ['Sales in thousands', 'LinReg', 'kNN', 'SVM', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGB']
prediction['Stacking'] = np.zeros(prediction.shape[0])
for train_index, test_index in kf.split(prediction):
    linreg_s = LinearRegression().fit(prediction.iloc[train_index, 1:], prediction.iloc[train_index, 0])
    prediction.iloc[test_index, 8] = linreg_s.predict(prediction.iloc[test_index, 1:])
#prediction.sort_values('Sales in thousands', inplace=True)
prediction.head()
score = []
for i in range(1, 9):
    score += [((prediction.iloc[:, i] - prediction['Sales in thousands'])**2).mean()]
best_model = score.index(min(score))
pd.Series(score, index = ['LinReg', 'kNN', 'SVM', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGB', 'Stacking'])
fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(111)
plt.plot(np.arange(prediction.shape[0]), prediction['Sales in thousands'], label='True Value', linewidth=3)
for i in range(len(prediction.columns[1:])):
    if i != best_model:
        plt.plot(np.arange(prediction.shape[0]), prediction.iloc[:, i+1], label=prediction.columns[i+1]+'('+str(round(score[i], 2))+')', c='C'+str(i+1))
for i in range(len(prediction.columns[1:])):
    if i == best_model:
        plt.plot(np.arange(prediction.shape[0]), prediction.iloc[:, i+1], label=prediction.columns[i+1]+'('+str(round(score[i], 2))+')', linewidth=3, c='C'+str(i+1))
handles, labels = ax.get_legend_handles_labels()
handles = handles[:5] + [handles[-1]] + handles[5:-1]
plt.legend(handles=handles, loc=9, bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.title('Prediction Results', fontsize=16)
plt.xticks(fontsize=13)
plt.xlabel('Instances', fontsize=16)
plt.yticks(fontsize=13)
plt.ylabel('Sales in thousands', fontsize=16)
plt.xlim(0, prediction.shape[0]-1)
plt.grid()
#plt.legend(loc=2)
plt.show()
df[df['Sales in thousands']>=200].iloc[:, :15]