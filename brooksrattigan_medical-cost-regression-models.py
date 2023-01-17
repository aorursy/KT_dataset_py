# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# For data visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns; sns.set()

# Disabling warnings
import warnings
warnings.simplefilter("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/insurance/insurance.csv")
df = data.copy()
display(df.head())
display(df.tail())
df.info()
df.describe().T
sns.swarmplot(x="sex", y="charges", data=df)
plt.title('Medical Costs by Gender', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="smoker", y="charges", data=df)
plt.title('Medical Costs by Smoking', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="smoker", y="bmi", data=df)
plt.title('Bmi by Smoking', color = 'blue', fontsize=15)
plt.show()
sns.boxplot(x="sex", y="charges",hue="smoker", data=df)
plt.title('Medical Costs by Gender and Smoking', color = 'blue', fontsize=15)
plt.show()

sns.boxplot(x="smoker", y="bmi", data=df)
plt.title('Bmi by Smoking', color = 'blue', fontsize=15)
plt.show()
df['sex'] = [1 if each=='male' else 0 for each in df.sex]
df['smoker']=[1 if each=='yes' else 0 for each in df.smoker]
df = pd.get_dummies(df, columns = ["region"], prefix = ["region"], drop_first=False)
df.head()
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor()
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
# When we sort the outlier scores, it is seen that the top 5 values seem to outstand compared to the rest.
# In this framework, the 6th observation is selected as the threshold and the outliers will be replaced by this threshold.
np.sort(df_scores)[:20]
# The threshold score is set as a filter.
threshold_score = np.sort(df_scores)[5]
df[df_scores == threshold_score]
outlier_tf = df_scores < threshold_score
# the threshold observation
threshold_observation = df[df_scores == threshold_score]
threshold_observation
outliers = df[outlier_tf]
outliers
# We convert these into an array to handle the outliers with to_records() method.
res = outliers.to_records(index = False)
res
# We replace the values of outlier observations with the values of the threshold observation. 
res[:] = threshold_observation.to_records(index = False)
df[outlier_tf] = pd.DataFrame(res, index = df[outlier_tf].index)
df[outlier_tf]
df.head()
import statsmodels.api as sm 
import statsmodels.formula.api as smf 
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict 
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
X = df.drop(["charges"],axis = 1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 42)
# Multilinear Regression model with skilearn 
lr = LinearRegression()
model1 = lr.fit(X_train, y_train)
# Coefficients
model1.coef_ 
# Intercept
model1.intercept_
# R2 score of test data
model1.score(X_test,y_test)
# RMSE score of test data
rmse = np.sqrt(mean_squared_error(y_test, model1.predict(X_test)))
rmse
y_pred = model1.predict(X_test)
y_test_ =np.array(range(0,len(y_test)))
plt.plot(y_test_,y_test,color="r")
plt.plot(y_test_,y_pred,color="blue")
plt.show()
# R2 average of test data after cross validation
mlin_final_r2 = cross_val_score(model1, X_train, y_train, cv = 10, scoring = "r2").mean()
mlin_final_r2
# RMSE average score of test data after cross validation
mlin_final_rmse = np.sqrt(-cross_val_score(model1, 
                X_test, 
                y_test, 
                cv = 10, 
                scoring = "neg_mean_squared_error")).mean()
mlin_final_rmse
from sklearn.cross_decomposition import PLSRegression, PLSSVD
pls_model = PLSRegression().fit(X_train, y_train)
# PLS model coefficients
pls_model.coef_
# PLS model predictions based on train data
y_pred = pls_model.predict(X_train)
# PLS RMSE score for train data
np.sqrt(mean_squared_error(y_train, y_pred))
# PLS R2 for train data
r2_score(y_train, y_pred)
# PLS prediction based on test data
y_pred = pls_model.predict(X_test)
# PLS RMSE test score
np.sqrt(mean_squared_error(y_test, y_pred))
# PLS R2 for test data
r2_score(y_test, y_pred)
# Illustraion of change in RMSE score as the model adds one additional component to the model in each loop.
cv_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

RMSE = []

for i in np.arange(1, X_train.shape[1] + 1):
    pls = PLSRegression(n_components=i)
    score = np.sqrt(-1*cross_val_score(pls, X_train, y_train, cv=cv_10, scoring='neg_mean_squared_error').mean())
    RMSE.append(score)

plt.plot(np.arange(1, X_train.shape[1] + 1), np.array(RMSE), '-v', c = "r")
plt.xlabel('Number of Components')
plt.ylabel('RMSE')
plt.title('Components and RMSE');
# PLS model with two components
pls_model2 = PLSRegression(n_components = 2).fit(X_train, y_train)
# PLS prediction based on test data after cross validation
y_pred2 = pls_model2.predict(X_test)
# PLS RMSE test score after cross validation
pls_final_rmse = np.sqrt(mean_squared_error(y_test, y_pred2))
pls_final_rmse
# PLS R2 test score after cross validation
pls_final_r2 = r2_score(y_test, y_pred2)
pls_final_r2
from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha = 0.1).fit(X_train, y_train)
ridge_model
ridge_model.coef_
# Illustration of how weights of independent variables approaches to 0 as the alpha value increases. 

lambdas = 10**np.linspace(10,-2,100)*0.5

ridge_model = Ridge()
coefficients = []

for i in lambdas:
    ridge_model.set_params(alpha = i)
    ridge_model.fit(X_train, y_train) 
    coefficients.append(ridge_model.coef_)
        
ax = plt.gca()
ax.plot(lambdas, coefficients) 
ax.set_xscale('log') 

plt.xlabel('Lambda(Alpha) Values')
plt.ylabel('Coefficients')
plt.title('Ridge Coefficients');
# Ridge prediction based on test data
y_pred = ridge_model.predict(X_test)
# Ridge RMSE test score
np.sqrt(mean_squared_error(y_test, y_pred))
# Ridge R2 
r2_score(y_test, y_pred)
from sklearn.linear_model import RidgeCV
# Ridge instantiation of cross validation model and model details
ridge_cv = RidgeCV(alphas = lambdas, 
                   scoring = "neg_mean_squared_error",
                   normalize = True)
ridge_cv.fit(X_train, y_train)
ridge_model
# Ridge cross validation alpha score
ridge_cv.alpha_
# Ridge tuned model after cross validation
ridge_tuned = Ridge(alpha = ridge_cv.alpha_, 
                   normalize = True).fit(X_train,y_train)
# Ridge model coefficients after cross validation
ridge_tuned.coef_
# Ridge RMSE test score after cross validation
ridge_final_rmse = np.sqrt(mean_squared_error(y_test, ridge_tuned.predict(X_test)))
ridge_final_rmse
# Ridge R2 after cross validation
ridge_final_r2 = r2_score(y_test, ridge_tuned.predict(X_test))
ridge_final_r2
from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha = 1.0).fit(X_train, y_train)
lasso_model
# Lasso model coefficients
lasso_model.coef_
# The weight of independent variables comes to value of zero as the alpha score changes. 

lasso = Lasso()
lambdas = 10**np.linspace(10,-2,100)*0.5 
coefficients = []

for i in lambdas:
    lasso.set_params(alpha=i)
    lasso.fit(X_train, y_train)
    coefficients.append(lasso.coef_)
    
ax = plt.gca()
ax.plot(lambdas*2, coefficients)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
# Lasso model prediction based on test data
y_pred = lasso_model.predict(X_test)
# Lasso RMSE test score
np.sqrt(mean_squared_error(y_test, y_pred))
# Lasso R2
r2_score(y_test, y_pred)
from sklearn.linear_model import LassoCV
lasso_cv_model = LassoCV(alphas = None, 
                         cv = 10, 
                         max_iter = 10000, 
                         normalize = True)
# Lasso cross validation model details
lasso_cv_model.fit(X_train,y_train)
# Lasso cross validation model alpha score
lasso_cv_model.alpha_
# Lasso tuned model after cross validation
lasso_tuned = Lasso(alpha = lasso_cv_model.alpha_)
lasso_tuned.fit(X_train, y_train)
# Lasso predictions of tuned model base on test data
y_pred = lasso_tuned.predict(X_test)
# Lasso model coefficients after cross validation
lasso_tuned.coef_
# Lasso RMSE test score after cross validation
lasso_final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
lasso_final_rmse
# Lasso R2 after cross validation
lasso_final_r2 = r2_score(y_test, y_pred)
lasso_final_r2
print(f"""Multilinear Regression RMSE: {mlin_final_rmse}, R2: {mlin_final_r2}
PLS Regression RMSE: {pls_final_rmse}, R2: {pls_final_r2}
Ridge Regression RMSE: {ridge_final_rmse}, R2: {ridge_final_r2}
Lasso Regression RMSE: {lasso_final_rmse}, R2: {lasso_final_r2}""")
from sklearn.preprocessing import PolynomialFeatures
# we can change the degree value for model tuning, but it is worthwhile to note that higher degree levels may lead to overfitting.
poly_features = PolynomialFeatures(degree=3)
x_train_poly = poly_features.fit_transform(X_train)
poly_model = LinearRegression()
poly_model.fit(x_train_poly, y_train)
y_train_pred = poly_model.predict(x_train_poly)
# Polynomial Regression RMSE and R2 score for train data
rmse_train = np.sqrt(mean_squared_error(y_train,y_train_pred))
r2_train = r2_score(y_train, y_train_pred)
print(rmse_train,r2_train)
y_test_pred = poly_model.predict(poly_features.fit_transform(X_test))
# Polynomial Regression RMSE and R2 score for test data
poly_rmse_final = np.sqrt(mean_squared_error(y_test, y_test_pred))
poly_r2_final = r2_score(y_test, y_test_pred)
print(poly_rmse_final,poly_r2_final)
y_test_ =np.array(range(0,len(y_test_pred)))
plt.plot(y_test_,y_test,color="r")
plt.plot(y_test_,y_test_pred,color="blue")
plt.show()
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
rf_model = RandomForestRegressor(random_state = 42)
rf_model.fit(X_train, y_train)

rf_model.predict(X_test)[0:5]
y_pred = rf_model.predict(X_test)
# RF RMSE test score
np.sqrt(mean_squared_error(y_test, y_pred))
# RF R2 score for test data
r2_score(y_test, y_pred)
rf_params = {'max_depth': list(range(1,10)),
            'max_features': [2,3,5,7],
            'n_estimators' : [100, 200, 500, 1000, 1500]}
rf_model = RandomForestRegressor(random_state = 42)
rf_cv_model = GridSearchCV(rf_model, 
                           rf_params, 
                           cv = 10, 
                            n_jobs = -1,
                            verbose = 2)
rf_cv_model.fit(X_train, y_train)
rf_cv_model.best_params_
rf_tuned = RandomForestRegressor(max_depth  = 5, 
                                 max_features = 7, 
                                 n_estimators =1000)
rf_tuned.fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)
# RF RMSE test score after model tuning
rf_rmse_final = np.sqrt(mean_squared_error(y_test, y_pred))
rf_rmse_final
# RF R2 for test data after model tuning
rf_r2_final = r2_score(y_test, y_pred)
rf_r2_final
# Importance level of independent variables through RF
Importance = pd.DataFrame({"Importance": rf_tuned.feature_importances_*100},
                         index = X_train.columns)
Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = True).plot(kind ="barh", color = "r")

plt.xlabel("Importance Levels of Variables")
import xgboost as xgb
DM_train = xgb.DMatrix(data = X_train, label = y_train)
DM_test = xgb.DMatrix(data = X_test, label = y_test)
from xgboost import XGBRegressor
xgb_model = XGBRegressor().fit(X_train, y_train)

# XGBoost RMSE test score
y_pred = xgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# XGBoost R2 for test data
r2_score(y_test, y_pred)
xgb_model
xgb_grid = {
     'colsample_bytree': [0.4, 0.5,0.6,0.9,1], 
     'n_estimators':[100, 200, 500, 1000],
     'max_depth': [2,3,4,5,6],
     'learning_rate': [0.1, 0.01, 0.5]
}
xgb = XGBRegressor()

xgb_cv = GridSearchCV(xgb, 
                      param_grid = xgb_grid, 
                      cv = 10, 
                      n_jobs = -1,
                      verbose = 2)


xgb_cv.fit(X_train, y_train)
xgb_cv.best_params_
xgb_tuned = XGBRegressor(colsample_bytree = 0.9, 
                         learning_rate = 0.01, 
                         max_depth = 3, 
                         n_estimators = 500) 

xgb_tuned = xgb_tuned.fit(X_train,y_train)
# XGBoost RMSE test score after model tuning
y_pred = xgb_tuned.predict(X_test)
xg_rmse_final = np.sqrt(mean_squared_error(y_test, y_pred))
xg_rmse_final
# XGBoost R2 after model tuning
xg_r2_final = r2_score(y_test, y_pred)
xg_r2_final
from lightgbm import LGBMRegressor
lgbm = LGBMRegressor()
lgbm_model = lgbm.fit(X_train, y_train)
y_pred = lgbm_model.predict(X_test, 
                            num_iteration = lgbm_model.best_iteration_)
# LGBM RMSE test score
np.sqrt(mean_squared_error(y_test, y_pred))
# LGBM R2 for test data
r2_score(y_test, y_pred)
lgbm_model
lgbm_grid = {
    'colsample_bytree': [0.4, 0.5,0.6,0.9,1],
    'learning_rate': [0.01, 0.1, 0.5,1],
    'n_estimators': [20, 40, 100, 200, 500,1000],
    'max_depth': [1,2,3,4,5,6,7,8] }

lgbm = LGBMRegressor()
lgbm_cv_model = GridSearchCV(lgbm, lgbm_grid, cv=10, n_jobs = -1, verbose = 2)
lgbm_cv_model.fit(X_train, y_train)
lgbm_cv_model.best_params_
lgbm_tuned = LGBMRegressor(learning_rate = 0.01, 
                           max_depth = 3, 
                           n_estimators = 500,
                          colsample_bytree = 0.9)

lgbm_tuned = lgbm_tuned.fit(X_train,y_train)
y_pred = lgbm_tuned.predict(X_test)
# LGBM RMSE test score after model tuning
lgbm_rmse_final = np.sqrt(mean_squared_error(y_test, y_pred))
lgbm_rmse_final
# LGBM R2 for test data after model tuning
lgbm_r2_final = r2_score(y_test, y_pred)
lgbm_r2_final
print(f"""Polynomial Regression RMSE: {poly_rmse_final}, R2: {poly_r2_final}
RF Regression RMSE: {rf_rmse_final}, R2: {rf_r2_final}
XGBoost Regression RMSE: {xg_rmse_final}, R2: {xg_r2_final}
LightGBM Regression RMSE: {lgbm_rmse_final}, R2: {lgbm_r2_final}""")
