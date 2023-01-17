!pip install tensorflow
!pip install keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error,r2_score
from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import metrics
from keras.wrappers.scikit_learn import KerasRegressor


data = pd.read_csv("../input/predict-sp500-monthly-return/timeseries_train.csv", sep = ";")
data
data.columns
#I rank of percentage of NaN values for each columns 
all_data_na = (data.isnull().sum() / len(data)) * 100
all_data_na = all_data_na.sort_values(ascending=False)[:163]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(163)

columns_to_drop = data.loc[:, np.sum(data.isnull()) > 120].columns
columns_to_drop
data = data.drop(columns_to_drop, axis = 1) 
print("dimension after dropping: {}".format(data.shape))
# I drop this column too, since in the test set has all nan values
data.drop('csp', axis = 1, inplace = True)
data.shape
NaN = data.loc[:,data.isnull().mean() > 0*100].columns # columns with at least one NaN
NaN = NaN.to_list()
NaN # 15 of the remaining columns have at least 1 NaN

data.loc[:, np.sum(data.isnull()) >= 1]
data.describe()
data.describe()
#fill the NaN using the mean
data.fillna(data.mean(), inplace = True)
# verify if the trasfrom function worked and if still there are any miss values  
data.isnull().any().sum()
d_cont = data.select_dtypes(include = 'float64')
d_cat = data.select_dtypes(exclude = 'float64')
d_cont
d_cat
data['HWI'] = data['HWI'].astype(float)

x = data.select_dtypes(include = 'float64')

#useful for plot
data['dates'] = pd.to_datetime(data['dates'])


y = data['CRSP_SPvw']
x.drop('CRSP_SPvw', axis = 1, inplace = True)
x
# top 10 variables correlated with the target features 'CRSP_SPvw'
cor = data.corr()

cor.sort_values('CRSP_SPvw', inplace = True, ascending = False)

cor['CRSP_SPvw'][1:20]
#top 10 var- corr negative ones
cor.sort_values('CRSP_SPvw', inplace = True, ascending = True)
cor['CRSP_SPvw'][0:10]
sc = StandardScaler()

x_scaled = sc.fit_transform(x)

x_scaled_train = x_scaled[:408]
y_train = y[:408]

x_scaled_test = x_scaled[408:]
y_test = y[408:]


alphas = np.logspace(-2, -.5, 20) 
params = [{'alpha': alphas}]

tss = TimeSeriesSplit()

ridge = Ridge(random_state=90, max_iter=1000)

grid_rid = GridSearchCV(ridge, params, cv=tss, scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error'], refit = 'neg_mean_squared_error', n_jobs = -1)

grid_rid.fit(x_scaled_train, y_train)

y_pred = grid_rid.predict(x_scaled_test)



plt.figure().set_size_inches(15, 6)
line_pred, = plt.plot(data.iloc[408:,0], y_pred, color = 'red', lw = 2)
line_true, = plt.plot(data.iloc[408:,0], y_test, lw = 2)
plt.grid()

plt.ylabel('SP500 monthly return rate')
plt.xlabel('Date')
plt.title('Prediction')
plt.legend([line_pred, line_true], ['Predicted', 'Real'])
MSE_rid = grid_rid.cv_results_['mean_test_neg_mean_squared_error']
MSE_std_rid = grid_rid.cv_results_['std_test_neg_mean_squared_error']

MAE_rid = grid_rid.cv_results_['mean_test_neg_mean_absolute_error']
MAE_std_rid = grid_rid.cv_results_['std_test_neg_mean_absolute_error']

plt.figure().set_size_inches(15, 6)

ax = plt.subplot(1,2,1)
plt.semilogx(alphas, MSE_rid)
std_error = MSE_std_rid/ np.sqrt(5) # 5 because TimeSeriesSplit has as default value n_folds = 5

plt.semilogx(alphas, MSE_rid + std_error, 'b--')
plt.semilogx(alphas, MSE_rid - std_error, 'b--')

plt.fill_between(alphas, MSE_rid + std_error, MSE_rid - std_error, alpha=0.2)

plt.ylabel('MSE +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(MSE_rid), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]])
plt.title("MSE")


ax = plt.subplot(1,2,2)
plt.semilogx(alphas, MAE_rid)
std_error = MAE_std_rid/ np.sqrt(5)

plt.semilogx(alphas, MAE_rid + std_error, 'b--')
plt.semilogx(alphas, MAE_rid - std_error, 'b--')

plt.fill_between(alphas, MAE_rid + std_error, MAE_rid - std_error, alpha=0.2)

plt.ylabel('MAE +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(MAE_rid), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]])
plt.title("MAE")
print(grid_rid.cv_results_['rank_test_neg_mean_squared_error'][0])
print(grid_rid.cv_results_['rank_test_neg_mean_absolute_error'][0])
# 19 and 0 comes from the rank returned by 'grid_rid.cv_results_'
print("Best parameter alpha:%s, MSE=%s, MAE=%s" % (grid_rid.best_params_['alpha'], -MSE_rid[19], -MAE_rid[0] ))
alphas = np.logspace(-5, -3.5, 20) 
params = [{'alpha': alphas}]

lasso = Lasso(random_state=90, max_iter=1000)
grid_las = GridSearchCV(lasso, params, cv=tss,scoring = ['neg_mean_squared_error','neg_mean_absolute_error'], refit= 'neg_mean_squared_error', n_jobs = -1)
grid_las.fit(x_scaled_train, y_train)

y_pred = grid_las.predict(x_scaled_test)



plt.figure().set_size_inches(15, 6)
line_pred, = plt.plot(data.iloc[408:,0], y_pred, color = 'red', lw = 2)
line_true, = plt.plot(data.iloc[408:,0], y_test, lw = 2)
plt.grid()

plt.ylabel('SP500 monthly return rate')
plt.xlabel('Date')
plt.title('Prediction')
plt.legend([line_pred, line_true], ['Predicted', 'Real'])
MSE_las = grid_las.cv_results_['mean_test_neg_mean_squared_error']
MSE_std_las = grid_las.cv_results_['std_test_neg_mean_squared_error']

MAE_las = grid_las.cv_results_['mean_test_neg_mean_absolute_error']
MAE_std_las = grid_las.cv_results_['std_test_neg_mean_absolute_error']

plt.figure().set_size_inches(15, 6)

ax = plt.subplot(1,2,1)
plt.semilogx(alphas, MSE_las)
std_error = MSE_std_las/ np.sqrt(5) # 5 because TimeSeriesSplit has as default value n_folds = 5

plt.semilogx(alphas, MSE_las + std_error, 'b--')
plt.semilogx(alphas, MSE_las - std_error, 'b--')

plt.fill_between(alphas, MSE_las + std_error, MSE_las - std_error, alpha=0.2)

plt.ylabel('MSE +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(MSE_las), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]])
plt.title("MSE")


ax = plt.subplot(1,2,2)
plt.semilogx(alphas, MAE_las)
std_error = MAE_std_las/ np.sqrt(5)

plt.semilogx(alphas, MAE_las + std_error, 'b--')
plt.semilogx(alphas, MAE_las - std_error, 'b--')

plt.fill_between(alphas, MAE_las + std_error, MAE_las - std_error, alpha=0.2)

plt.ylabel('MAE +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(MAE_las), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]])
plt.title("MAE")
print(grid_las.cv_results_['rank_test_neg_mean_squared_error'][0])
print(grid_las.cv_results_['rank_test_neg_mean_absolute_error'][0])
# 19 and 19 comes from the rank returned by 'grid_rid.cv_results_'
print("Best parameter alpha:%s, MSE=%s, MAE=%s" % (grid_las.best_params_['alpha'], -MSE_las[19], -MAE_las[19] ))
poly = PolynomialFeatures(2)
X_poly = poly.fit_transform(x_scaled)
x_scaled_train = X_poly[:408]


x_scaled_test = X_poly[408:]
alphas = np.logspace(-5, -3.5, 20) 
params = [{'alpha': alphas}]

lasso = Lasso(random_state=90, max_iter=1000)


grid_poly = GridSearchCV(lasso, params, cv = tss,scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error'], refit = 'neg_mean_squared_error', n_jobs = -1)
grid_poly.fit(x_scaled_train, y_train)

y_pred = grid_poly.predict(x_scaled_test)



plt.figure().set_size_inches(15, 6)
line_pred, = plt.plot(data.iloc[408:,0], y_pred, color = 'red', lw = 2)
line_true, = plt.plot(data.iloc[408:,0], y_test, lw = 2)
plt.grid()

plt.ylabel('SP500 monthly return rate')
plt.xlabel('Date')
plt.title('Prediction')
plt.legend([line_pred, line_true], ['Predicted', 'Real'])
print(grid_poly.cv_results_['rank_test_neg_mean_squared_error'][0])
print(grid_poly.cv_results_['rank_test_neg_mean_absolute_error'][0])
MSE_poly = grid_poly.cv_results_['mean_test_neg_mean_squared_error']
MAE_poly = grid_poly.cv_results_['mean_test_neg_mean_absolute_error']


print("Best parameter:{}, MSE={:.5f}, MAE={:.5f}".format(grid_poly.best_params_, -MSE_poly[14], -MAE_poly[8]))
rf = RandomForestRegressor(n_estimators = 100, oob_score = True) 
params = {'max_depth': [6,8], 'min_samples_split': [12,15,18], 'min_samples_leaf': [2,3,6]}

grid_rf = GridSearchCV(rf, param_grid = params, cv = tss, scoring = ['neg_mean_squared_error','neg_mean_absolute_error' ], refit = 'neg_mean_squared_error',n_jobs = -1)

grid_rf.fit(x_scaled_train,y_train)
y_pred_rf = grid_rf.predict(x_scaled_test)



plt.figure().set_size_inches(15, 6)
line_pred, = plt.plot(data.iloc[408:,0], y_pred_rf, color = 'red', lw = 2)
line_true, = plt.plot(data.iloc[408:,0], y_test, lw = 2)
plt.grid()

plt.ylabel('SP500 monthly return rate')
plt.xlabel('Date')
plt.title('Prediction')
plt.legend([line_pred, line_true], ['Predicted', 'Real'])


print(grid_rf.cv_results_['rank_test_neg_mean_squared_error'][0])
print(grid_rf.cv_results_['rank_test_neg_mean_absolute_error'][0])
MSE_rf = grid_rf.cv_results_['mean_test_neg_mean_squared_error']
MAE_rf = grid_rf.cv_results_['mean_test_neg_mean_absolute_error']


print("Best parameter:{}, MSE={:.5f}, MAE={:.5f}".format(grid_rf.best_params_, -MSE_rf[7], -MAE_rf[7]))

data_all = pd.read_csv("../input/predict-sp500-monthly-return/timeseries_all.csv", sep = ";")

data_test = pd.read_csv("../input/predict-sp500-monthly-return/timeseries_test.csv", sep = ";")
data_test['csp']
# dropping the same columns as the train set
data_test = data_test.drop(columns_to_drop, axis = 1)
# dropping also the target variable and this'csp' that has all NaNs 
data_test = data_test.drop(["Unnamed: 162", "csp"], axis = 1)
# filling with the mean
data_test.fillna(data_test.mean(), inplace = True)
data_test.isna().any().sum()
data_test.shape
data_test


data_test['dates']
data_test['HWI'] = data_test['HWI'].astype(float)

x_TEST = data.select_dtypes(include = 'float64')

#useful for plot
data['dates'] = pd.to_datetime(data_test['dates'])

x_TEST = data_test.select_dtypes(include = 'float64')

# scaling
x_TEST_scaled = sc.fit_transform(x_TEST)

# transforming for polynomial
x_TEST_poly = poly.fit_transform(x_TEST_scaled)

# selecting the target variable to compare
y_TEST = data_all.iloc[480:603, -1]
x_TEST_scaled.shape

y_TEST.shape
# gridsearch with ridge
y_pred_gr = grid_rid.predict(x_TEST_scaled)
mse_gr = mean_squared_error(y_TEST, y_pred_gr)

# gridsearch with lasso
y_pred_gl = grid_las.predict(x_TEST_scaled)
mse_gl = mean_squared_error(y_TEST, y_pred_gl)
#gridsearch with polynomial features
y_pred_gp = grid_poly.predict(x_TEST_poly)
mse_gp = mean_squared_error(y_TEST, y_pred_gp)
# gridsearch with randomforest 
y_pred_grf = grid_rf.predict(x_TEST_scaled)
mse_grf = mean_squared_error(y_TEST, y_pred_grf)
print("MSE(GridSearch with Ridge): {:>27.6f}".format(mse_gr))
print("MSE(GridSearch with Lasso): {:>27.6f}".format(mse_gl))
print("MSE(GridSearch with PolyFeatures): {:.6f}".format(mse_gr))
print("MSE(GridSearch with RandomForest): {:>20.6f}".format(mse_grf))

data_test['dates'] = pd.to_datetime(data_test['dates'])
plt.figure().set_size_inches(15, 6)
line_pred, = plt.plot(data_test['dates'], y_pred_grf, color = 'red', lw = 5)
line_true, = plt.plot(data_test['dates'], y_TEST, lw = 5)
plt.grid()

plt.ylabel('SP500 monthly return rate')
plt.xlabel('Date')
plt.title('Prediction')
plt.legend([line_pred, line_true], ['Predicted', 'Real'])
