import numpy as np

import pandas as pd

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/electric-motor-temperature/pmsm_temperature_data.csv')
df_test = df[(df['profile_id'] == 65) | (df['profile_id'] == 72)]

df = df[(df['profile_id'] != 65) & (df['profile_id'] != 72)]
df
df.describe()
df.isnull().sum()
plt.figure(figsize=(15,6))

df['profile_id'].value_counts().sort_values().plot(kind = 'bar')
for i in df.columns:

    sns.distplot(df[i],color='g')

    sns.boxplot(df[i],color = 'y')

    plt.vlines(df[i].mean(),ymin = -1,ymax = 1,color = 'r')

    plt.show()
import scipy.stats as stats

for i in df.columns:

    print(i,' :\nSkew : ',df[i].skew(),' : \nKurtosis : ',df[i].kurt())

    print()
plt.figure(figsize=(14,7))

sns.heatmap(df.corr(),annot=True)
plt.figure(figsize=(20,5))

df[df['profile_id'] == 20]['stator_yoke'].plot(label = 'stator yoke')

df[df['profile_id'] == 20]['stator_tooth'].plot(label = 'stator tooth')

df[df['profile_id'] == 20]['stator_winding'].plot(label = 'stator winding')

plt.legend()
df.drop('profile_id',axis = 1,inplace=True)

df_test.drop('profile_id',axis = 1,inplace=True)
sns.distplot(df['ambient'])
from scipy.stats import shapiro

shapiro(df['ambient'])
shapiro(df['pm'])
from scipy.stats import bartlett

bartlett(df['ambient'],df['pm'])
sns.distplot(df['coolant'])
from scipy.stats import shapiro

shapiro(df['coolant'])
shapiro(df['pm'])
from scipy.stats import bartlett

bartlett(df['coolant'],df['pm'])
sns.distplot(df['u_d'])
from scipy.stats import shapiro

shapiro(df['u_d'])
shapiro(df['pm'])
from scipy.stats import bartlett

bartlett(df['u_d'],df['pm'])
sns.distplot(df['u_q'])
from scipy.stats import shapiro

shapiro(df['u_q'])
shapiro(df['pm'])
from scipy.stats import bartlett

bartlett(df['u_q'],df['pm'])
sns.distplot(df['motor_speed'])
from scipy.stats import shapiro

shapiro(df['motor_speed'])
shapiro(df['pm'])
from scipy.stats import bartlett

bartlett(df['motor_speed'],df['pm'])
sns.distplot(df['i_d'])
from scipy.stats import shapiro

shapiro(df['i_d'])
shapiro(df['pm'])
from scipy.stats import bartlett

bartlett(df['i_d'],df['pm'])
sns.distplot(df['i_q'])
from scipy.stats import shapiro

shapiro(df['i_q'])
shapiro(df['pm'])
from scipy.stats import bartlett

bartlett(df['i_q'],df['pm'])
df = df.sample(frac=1,random_state=3)
df.head()
sns.scatterplot(df['ambient'],df['pm'])
sns.scatterplot(df['coolant'],df['pm'])
sns.scatterplot(df['motor_speed'],df['pm'])
sns.scatterplot(df['u_q'],df['pm'])
sns.scatterplot(df['u_d'],df['pm'])
sns.scatterplot(df['i_q'],df['pm'])
sns.scatterplot(df['i_d'],df['pm'])
from sklearn.preprocessing import MinMaxScaler

X = df.drop(['pm','stator_yoke','stator_tooth','stator_winding','torque'],axis = 1)

X_df_test = df_test.drop(['pm','stator_yoke','stator_tooth','stator_winding','torque'],axis = 1)

mm = MinMaxScaler()

X = mm.fit_transform(X)

X_df_test = mm.fit_transform(X_df_test)

y = df['pm']

y_df_test = df_test['pm']

X = pd.DataFrame(X,columns = ['ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'i_d','i_q'])

X_df_test = pd.DataFrame(X_df_test,columns = ['ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'i_d','i_q'])

y.reset_index(drop = True,inplace = True)

y_df_test.reset_index(drop = True,inplace = True)
print(X.shape)

print(y.shape)
for i in X.columns:

    print(X[i].skew())

    sns.distplot(X[i],color='g')

    sns.boxplot(X[i],color = 'y')

    plt.vlines(X[i].mean(),ymin = -1,ymax = 1,color = 'r')

    plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
import statsmodels.api as sm

X_train_const = sm.add_constant(X_train)

lin_reg = sm.OLS(y_train,X_train_const).fit()

lin_reg.summary()
from statsmodels.stats.diagnostic import linear_rainbow

linear_rainbow(lin_reg)
from statsmodels.stats.api import het_goldfeldquandt

het_goldfeldquandt(lin_reg.resid,lin_reg.model.exog)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = [variance_inflation_factor(X_train_const.values,i) for i in range(X_train_const.shape[1])]

pd.DataFrame(vif,index=X_train_const.columns)
lin_reg.resid.plot(kind = 'density')
import scipy.stats as stats

import pylab

st_residual = lin_reg.get_influence().resid_studentized_internal

stats.probplot(st_residual, dist="norm", plot = pylab)

plt.show()
y_train_pred = lin_reg.predict(X_train_const)

train_rmse = np.sqrt(np.sum(((y_train-y_train_pred)**2))/len(y_train))

train_rmse
X_test_const = sm.add_constant(X_test)

y_test_pred = lin_reg.predict(X_test_const)

y_test_pred
test_rmse = np.sqrt(np.sum(((y_test-y_test_pred)**2))/len(y_test))

test_rmse
lin_reg.rsquared_adj
X_trans = X

X_trans['coolant'] = np.power(X_trans['coolant'],1/3)

X_trans['ambient'] = np.power(X_trans['ambient'],3)

X_trans['i_d'] = np.power(X_trans['i_d'],3)
for i in X_trans.columns:

    print(X_trans[i].skew())

    sns.distplot(X_trans[i],color='g')

    sns.boxplot(X_trans[i],color = 'y')

    plt.vlines(X_trans[i].mean(),ymin = -1,ymax = 1,color = 'r')

    plt.show()
z = np.abs(stats.zscore(X_trans))

print(z)
X_trans = X_trans.drop(np.where(z > 3)[0][0:])

X_trans.reset_index(drop=True,inplace = True)

y = y.drop(np.where(z > 3)[0][0:])

y.reset_index(drop = True,inplace = True)
print(X_trans.shape)

print(y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.3, random_state=3)
import statsmodels.api as sm

X_train_const = sm.add_constant(X_train)

lin_reg = sm.OLS(y_train,X_train_const).fit()

lin_reg.summary()
y_train_pred = lin_reg.predict(X_train_const)

train_rmse = np.sqrt(np.sum(((y_train-y_train_pred)**2))/len(y_train))

train_rmse
X_test_const = sm.add_constant(X_test)

y_test_pred = lin_reg.predict(X_test_const)

y_test_pred
test_rmse = np.sqrt(np.sum(((y_test-y_test_pred)**2))/len(y_test))

test_rmse
X = X_trans
from sklearn.decomposition import PCA

pca  = PCA()

pca.fit(X)
pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)
pca5 = PCA(n_components=5)

X_pca = pca5.fit_transform(X)

X_pca
X_pca_train, X_pca_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=3)
X_pca_train_const = sm.add_constant(X_pca_train)

lin_reg = sm.OLS(y_train,X_pca_train_const).fit()

lin_reg.summary()
y_train_pred = lin_reg.predict(X_pca_train_const)

train_rmse = np.sqrt(np.sum(((y_train-y_train_pred)**2))/len(y_train))

train_rmse
X_pca_test_const = sm.add_constant(X_pca_test)

y_test_pred = lin_reg.predict(X_pca_test_const)

y_test_pred
test_rmse = np.sqrt(np.sum(((y_test-y_test_pred)**2))/len(y_test))

test_rmse
X_wo_dqi = X.drop(['i_d','i_q'],axis = 1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_wo_dqi, y, test_size=0.3, random_state=3)
import statsmodels.api as sm

X_train_const = sm.add_constant(X_train)

lin_reg = sm.OLS(y_train,X_train_const).fit()

lin_reg.summary()
y_train_pred = lin_reg.predict(X_train_const)

train_rmse = np.sqrt(np.sum(((y_train-y_train_pred)**2))/len(y_train))

train_rmse
X_test_const = sm.add_constant(X_test)

y_test_pred = lin_reg.predict(X_test_const)

y_test_pred
test_rmse = np.sqrt(np.sum(((y_test-y_test_pred)**2))/len(y_test))

test_rmse
X_wo_ms = X.drop(['motor_speed'],axis = 1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_wo_ms, y, test_size=0.3, random_state=3)
import statsmodels.api as sm

X_train_const = sm.add_constant(X_train)

lin_reg = sm.OLS(y_train,X_train_const).fit()

lin_reg.summary()
y_train_pred = lin_reg.predict(X_train_const)

train_rmse = np.sqrt(np.sum(((y_train-y_train_pred)**2))/len(y_train))

train_rmse
X_test_const = sm.add_constant(X_test)

y_test_pred = lin_reg.predict(X_test_const)

y_test_pred
test_rmse = np.sqrt(np.sum(((y_test-y_test_pred)**2))/len(y_test))

test_rmse
y = pd.DataFrame(y)
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# lr = LinearRegression()

# ridge = Ridge(alpha = 20000)

lasso = Lasso(alpha = 0.012)
# from sklearn.model_selection import KFold

# from sklearn import metrics

# kf = KFold(n_splits=5,shuffle=True,random_state=0)

# for model,name in zip([lr,ridge,lasso],['LR','Ridge','Lasso']):

#     mse_li = []

#     for train_idx,test_idx in kf.split(X,y):

#         X_train,X_test = X.iloc[train_idx,:],X.iloc[test_idx,:]

#         y_train,y_test = y.iloc[train_idx,:],y.iloc[test_idx,:]

#         model.fit(X_train,y_train)

#         y_pred = model.predict(X_test)

#         mse = metrics.mean_squared_error(y_test,y_pred)

#         mse_li.append(mse)

#     print('RMSE scores : %0.03f (+/- %0.08f) [%s]'%(np.mean(mse_li), np.var(mse_li,ddof = 1), name))

#     print()
# from sklearn.ensemble import RandomForestRegressor

# from sklearn.tree import DecisionTreeRegressor

# from sklearn.model_selection import RandomizedSearchCV

# from scipy.stats import randint

# dt = DecisionTreeRegressor(random_state=0)

# rf = RandomForestRegressor(random_state=0,n_jobs = -1)

# param_dt = {

#         'criterion' : ['mse','mae'],

#         'max_depth' : randint(1,11)

# }

# param_rf = {

#         'n_estimators' : randint(1,70),

#         'max_depth' : randint(1,11)

# }

# rscv_dt = RandomizedSearchCV(dt,param_dt,scoring='neg_mean_squared_error',cv = 5,n_jobs=1,n_iter = 2,verbose = 1000,random_state = 0)

# rscv_rf = RandomizedSearchCV(rf,param_rf,scoring='neg_mean_squared_error',cv = 5,n_jobs=-1,n_iter = 2,verbose = 1000,random_state = 0)

# rscv_dt.fit(X,y)

# rscv_rf.fit(X,y)

# print(rscv_dt.best_params_)

# print(rscv_rf.best_params_)
# from sklearn.ensemble import RandomForestRegressor

# from sklearn.tree import DecisionTreeRegressor

# from sklearn.neighbors import KNeighborsRegressor

# dt = DecisionTreeRegressor(criterion='mse',max_depth=6,random_state=0)

# rf = RandomForestRegressor(n_estimators=41,max_depth=6,random_state=0,n_jobs = -1)
# from sklearn.model_selection import KFold

# from sklearn import metrics

# kf = KFold(n_splits=5,shuffle=True,random_state=0)

# for model,name in zip([dt,rf],['DT','RF']):

#     mse_li = []

#     for train_idx,test_idx in kf.split(X,y):

#         X_train,X_test = X.iloc[train_idx,:],X.iloc[test_idx,:]

#         y_train,y_test = y.iloc[train_idx,:],y.iloc[test_idx,:]

#         model.fit(X_train,y_train)

#         y_pred = model.predict(X_test)

#         mse = metrics.mean_squared_error(y_test,y_pred)

#         mse_li.append(mse)

#     print('RMSE scores : %0.03f (+/- %0.08f) [%s]'%(np.mean(mse_li), np.var(mse_li,ddof = 1), name))

#     print()
from sklearn.ensemble import BaggingRegressor

# from sklearn.model_selection import KFold, cross_val_score

# models = []

# models.append(("LinearRegression",lr))

# models.append(("Lasso",lasso))

# models.append(("Ridge",ridge))

# models.append(("DT",dt))

# for name,model in models:

#     mse_var = []

#     for val in np.arange(1,21):

#         bg_model = BaggingRegressor(base_estimator=model,n_estimators=val,n_jobs=-1,verbose = 1000, random_state = 0)

#         kfold = KFold(n_splits=5,shuffle=True,random_state=0)

#         results = cross_val_score(bg_model,X,y,cv=kfold,n_jobs=-1,scoring='neg_mean_squared_error',verbose = 1000)

#         mse_var.append(np.var(results,ddof = 1))

#     print(name,np.argmin(mse_var)+1)
# from sklearn.ensemble import AdaBoostRegressor

# from sklearn.model_selection import KFold, cross_val_score

# models = []

# models.append(("LinearRegression",lr))

# models.append(("Lasso",lasso))

# models.append(("Ridge",ridge))

# models.append(("DT",dt))

# models.append(("RF",rf))

# for name,model in models:

#     mse_mean = []

#     for val in np.arange(1,21):

#         bg_model = AdaBoostRegressor(base_estimator=model,n_estimators=val, random_state = 0)

#         kfold = KFold(n_splits=5,shuffle=True,random_state=0)

#         results = cross_val_score(bg_model,X,y,cv=kfold,n_jobs=-1,scoring='neg_mean_squared_error',verbose = 1000)

#         mse_mean.append(np.mean(results))

#     print(name,np.argmax(mse_mean)+1)
# #Bagging Models

# LR_bag = BaggingRegressor(base_estimator = lr,n_estimators = 12,random_state = 0,n_jobs = -1)

lasso_bag = BaggingRegressor(base_estimator = lasso,n_estimators = 2,random_state = 0,n_jobs = -1)

# DT_bag = BaggingRegressor(base_estimator = dt,n_estimators = 3,random_state = 0,n_jobs = -1,verbose = 1000)

# ridge_bag = BaggingRegressor(base_estimator = ridge,n_estimators = 2,random_state = 0,n_jobs = -1) 

# # #Boosting models

# lasso_boost = AdaBoostRegressor(base_estimator = lasso,n_estimators = 10,random_state = 0)

# ridge_boost = AdaBoostRegressor(base_estimator = ridge,n_estimators = 3,random_state = 0)

# DT_boost = AdaBoostRegressor(base_estimator = dt,n_estimators = 15,random_state = 0)

# RF_boost = AdaBoostRegressor(base_estimator = rf,n_estimators = 8,random_state = 0)
# from sklearn.ensemble import GradientBoostingRegressor

# GBC = GradientBoostingRegressor(n_estimators = 100,random_state = 0)
# models = []

# models.append(('LR Bagged',LR_bag))

# models.append(('Lasso Bagged',lasso_bag))

# models.append(('Lasso Boosted',lasso_boost))

# models.append(('Ridge Bagged',ridge_bag))

# models.append(('Ridge Boosted',ridge_boost))

# models.append(('DTree Bagged',DT_bag))

# models.append(('DTree Boosted',DT_boost))

# models.append(('Gradient Boost',GBC))

# models.append(('RF Boosted',RF_boost))
# results = []

# names = []

# for name, model in models:

#     kfold = KFold(n_splits = 5,random_state = 0,shuffle = True)

#     cv_results = cross_val_score(model,X,y,cv = kfold,scoring='neg_mean_squared_error',n_jobs = -1)

#     results.append(cv_results)

#     names.append(name)

#     print(name,' : ',np.mean(cv_results),' -- ',np.var(cv_results,ddof = 1))
from sklearn.metrics import r2_score,mean_squared_error

lasso_bag.fit(X,y)

test_pred = lasso_bag.predict(X_df_test)
test_pred