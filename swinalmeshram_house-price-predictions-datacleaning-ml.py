import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from scipy import stats

import statsmodels.api as sm





from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score,mean_squared_error

from sklearn.linear_model import Lasso,LassoCV,Ridge,RidgeCV,ElasticNet,ElasticNetCV

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

dft = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

# pd.set_option('display.max_rows',None)

# pd.set_option('display.max_columns',None)
df.head()
dft.head()
# checking for null Values

df.isnull().sum()
full = df.append(dft)
full.info()
df.shape
sns.distplot(df['SalePrice'],fit = stats.norm)
mu,sigma = stats.norm.fit(df['SalePrice'])
# mu, sigma
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

plt.ylabel('Frequency')



#Probablity plot



fig = plt.figure()

stats.probplot(df['SalePrice'], plot=plt)

plt.show()
sm.qqplot(df['SalePrice'],line = 'r')
df['SalePrice'] = np.log(df['SalePrice']+1)



#Check again for more normal distribution



plt.subplots(figsize=(12,9))

sns.distplot(df['SalePrice'], fit=stats.norm)



# Get the fitted parameters used by the function

(mu, sigma) = stats.norm.fit(df['SalePrice'])



# plot with the distribution



plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

plt.ylabel('Frequency')



#Probablity plot



fig = plt.figure()

stats.probplot(df['SalePrice'], plot=plt)

plt.show()

df.isnull().sum()
# % of missing values

Isnull  = df.isnull().sum()/len(df) * 100

Isnull = Isnull[Isnull>0]

Isnull.sort_values(inplace = True,ascending = False)

Isnull
Isnull = Isnull.to_frame()
Isnull.columns = ['count']

Isnull.index.names = ['Name']

Isnull['Name'] = Isnull.index
Isnull
#plot Missing values

plt.figure(figsize=(13, 5))

sns.set(style='whitegrid')

sns.barplot(x='Name', y='count', data=Isnull)

plt.xticks(rotation = 60)

plt.show()
df_corr = df.select_dtypes(include=[np.number])
df_corr.shape
df_corr
#Delete Id because that is not need for corralation plot

del df_corr['Id']
corr = df_corr.corr()

plt.subplots(figsize=(20,9))

sns.heatmap(corr, annot=True)
top_feature = corr.index[abs(corr['SalePrice']>0.5)]

plt.subplots(figsize=(12, 8))

top_corr = df[top_feature].corr()

sns.heatmap(top_corr, annot=True)

plt.show()
num_cols = list(df.select_dtypes(include = np.number))
num_colst = list(dft.select_dtypes(include = np.number))
dfn = df[num_cols]

dftn = dft[num_colst]
# dfn.isnull().sum()
# dftn.isnull().sum()
dfn['LotFrontage'].fillna(dfn['LotFrontage'].mode()[0],inplace = True)

dftn['LotFrontage'].fillna(dftn['LotFrontage'].mode()[0],inplace = True)
dfn['MasVnrArea'].fillna(dfn['MasVnrArea'].mode()[0],inplace = True)

dftn['MasVnrArea'].fillna(dftn['MasVnrArea'].mode()[0],inplace = True)

dfn['GarageYrBlt'].unique()
dfn['GarageYrBlt'].fillna(dfn['GarageYrBlt'].median(),inplace = True)

dftn['GarageYrBlt'].fillna(dftn['GarageYrBlt'].median(),inplace = True)



dfn.info()
dftn.isnull().sum()
dftn['BsmtFinSF1'].fillna(method='ffill',inplace = True)

dftn['BsmtFinSF2'].fillna(method='ffill',inplace = True)

dftn['BsmtUnfSF'].fillna(method='ffill',inplace = True)

dftn['TotalBsmtSF'].fillna(method='ffill',inplace = True)

dftn['BsmtFullBath'].fillna(method='ffill',inplace = True)

dftn['BsmtHalfBath'].fillna(method='ffill',inplace = True)

dftn['GarageCars'].fillna(method='ffill',inplace = True)

dftn['GarageArea'].fillna(method='ffill',inplace = True)
dftn.isnull().sum()
dfn['MasVnrArea'].unique()
dfn['MasVnrArea'] = dfn['MasVnrArea'].astype(float)

dfn.isnull().sum()
dfn.drop('Id',axis = 1,inplace = True)

dftn.drop('Id',axis = 1,inplace = True)

for col in dfn.columns:

    print(' ')

    print(col)

    print(dfn[col].value_counts())
co = list(dfn['OverallQual'].value_counts().head(5).index)

dfn['OverallQual'] = np.where(dfn['OverallQual'].isin(co),dfn['OverallQual'],'Other')

co = list(dftn['OverallQual'].value_counts().head(5).index)

dftn['OverallQual'] = np.where(dftn['OverallQual'].isin(co),dftn['OverallQual'],'Other')
co = list(dfn['OverallCond'].value_counts().head(3).index)

dfn['OverallCond'] = np.where(dfn['OverallCond'].isin(co),dfn['OverallCond'],'Other')



co = list(dftn['OverallCond'].value_counts().head(3).index)

dftn['OverallCond'] = np.where(dftn['OverallCond'].isin(co),dftn['OverallCond'],'Other')
cd = list(dfn['YearBuilt'].value_counts().head(10).index)

dfn['YearBuilt'] = np.where(dfn['YearBuilt'].isin(cd),dfn['YearBuilt'],'Other')



cd = list(dftn['YearBuilt'].value_counts().head(10).index)

dftn['YearBuilt'] = np.where(dftn['YearBuilt'].isin(cd),dftn['YearBuilt'],'Other')
cf = list(dfn['YearRemodAdd'].value_counts().head(10).index)

dfn['YearRemodAdd'] = np.where(dfn['YearRemodAdd'].isin(cf),dfn['YearRemodAdd'],'Other')

cf = list(dftn['YearRemodAdd'].value_counts().head(10).index)

dftn['YearRemodAdd'] = np.where(dftn['YearRemodAdd'].isin(cf),dftn['YearRemodAdd'],'Other')
ce = list(dfn['GarageYrBlt'].value_counts().head(10).index)

dfn['GarageYrBlt'] = np.where(dfn['GarageYrBlt'].isin(ce),dfn['GarageYrBlt'],'Other')



ce = list(dftn['GarageYrBlt'].value_counts().head(10).index)

dftn['GarageYrBlt'] = np.where(dftn['GarageYrBlt'].isin(ce),dftn['GarageYrBlt'],'Other')
cg = list(dfn['GarageCars'].value_counts().head(3).index)

dfn['GarageCars'] = np.where(dfn['GarageCars'].isin(cg),dfn['GarageCars'],'Other')



cg = list(dftn['GarageCars'].value_counts().head(3).index)

dftn['GarageCars'] = np.where(dftn['GarageCars'].isin(cg),dftn['GarageCars'],'Other')
cat_cols = ['OverallQual','OverallCond','YearBuilt','GarageYrBlt','GarageCars','YearRemodAdd']
dfn=pd.get_dummies(dfn,columns=cat_cols,drop_first=True)

dftn=pd.get_dummies(dftn,columns=cat_cols,drop_first=True)
print(dfn.shape)

print(dftn.shape)
X = dfn.drop('SalePrice',axis = 1)

y = dfn['SalePrice']
from sklearn.ensemble import RandomForestRegressor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state = 42)
rf = RandomForestRegressor()

rf = rf.fit(X_train, y_train)

y_pred =  rf.predict(X_test)

rf.score(X_test,y_test)
r2_score(y_test,y_pred)
print('RMSe : ',np.sqrt(mean_squared_error(y_test,y_pred)))
dfn.head()
dftn.head()
def rmsle(real, predicted):

    sum=0.0

    for x in range(len(predicted)):

        if predicted[x]<0 or real[x]<0: #check for negative values

            continue

        p = np.log(predicted[x]+1)

        r = np.log(real[x]+1)

        sum = sum + (p - r)**2

    return (sum/len(predicted))**0.5
X = dfn.drop('SalePrice',axis = 1)

y = dfn['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state = 42)
from xgboost import XGBRegressor
xgb = XGBRegressor()

xgb = xgb.fit(X_train,y_train)

y_pred = xgb.predict(X_test)
xgb.score(X_test,y_test)
r2_score(y_test,y_pred)
print('RMSe : ',np.sqrt(mean_squared_error(y_test,y_pred)))
# rmsle(y_test,y_pred)
X = dfn.drop('SalePrice',axis = 1)

y = dfn['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state = 42)
from sklearn.metrics import mean_squared_error

def adjusted_r2_score(y_true, y_pred, X_test):

    r2 = r2_score(y_true=y_true, y_pred=y_pred)

    adjusted_r2 = 1 - (1-r2)*(len(y_true)-1)/(len(y_true) - X_test.shape[1]-1)

    return adjusted_r2
xgr = XGBRegressor(objective='reg:linear', n_estimators=50, max_depth=5, n_jobs=-1, random_state=42)

xgr.fit(X_train, y_train)



y_pred = xgr.predict(X_test)



rsq_baseline2_xgb = r2_score(y_true=y_test, y_pred=y_pred)

adj_rsq_baseline2_xgb = adjusted_r2_score(y_true=y_test, y_pred=y_pred, X_test=X_test)

rmse_baseline2_xgb = mean_squared_error(y_true=y_test, y_pred=y_pred) ** 0.5

print('R-sq:', rsq_baseline2_xgb)

print('Adj. R-sq:', adj_rsq_baseline2_xgb)

print('RMSE:', rmse_baseline2_xgb)
from sklearn.tree import DecisionTreeRegressor
X = dfn.drop('SalePrice',axis = 1)

y = dfn['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state = 42)
dt = DecisionTreeRegressor()

dt = dt.fit(X_train,y_train)

y_pred = dt.predict(X_test)


print('r2_score: ',r2_score(y_test,y_pred))

print('Accuracy: ',dt.score(X_test,y_test))

print('RMSE: ',np.sqrt(mean_squared_error(y_test,y_pred)))

print('adj r sq.: ',1 - (1-r2_score(y_test,y_pred))*(len(y_test)-1)/(len(y_test) - X_test.shape[1]-1) )
from sklearn.model_selection import GridSearchCV

# # rf = RandomForestRegressor()



# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



# rfc = RandomForestRegressor(n_jobs=-1 , oob_score = True,random_state = 42) 



# param_grid = {

#     'n_estimators': [50,100,200, 700],

#     'max_features': ['auto', 'sqrt', 'log2'],

#     }



# CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)

# CV_rfc.fit(X_train,y_train)

# print('\n',CV_rfc.best_estimator_)
rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,

                      max_features='sqrt', max_leaf_nodes=None,

                      min_impurity_decrease=0.0, min_impurity_split=None,

                      min_samples_leaf=1, min_samples_split=2,

                      min_weight_fraction_leaf=0.0, n_estimators=700, n_jobs=-1,

                      oob_score=True, random_state=42, verbose=0,

                      warm_start=False)





X = dfn.drop('SalePrice',axis = 1)

y = dfn['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state = 42)



rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

# y_pred = np.exp(y_pred-1)

print('r2_score: ',r2_score(y_test,y_pred))

print('Accuracy: ',dt.score(X_test,y_test))

print('RMSE: ',np.sqrt(mean_squared_error(y_test,y_pred)))

print('adj r sq.: ',1 - (1-r2_score(y_test,y_pred))*(len(y_test)-1)/(len(y_test) - X_test.shape[1]-1) )
X.head()
dftn.head()
pred = rf.predict(dftn)
predantilog = np.exp(pred)-1
predantilog
import xgboost as xgb
data_dmatrix = xgb.DMatrix(data=X,label=y)
# X_train.head()
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,

                'max_depth': 20, 'alpha': 10}



cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,

                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
cv_results.head()
print((cv_results["test-rmse-mean"]).tail(1))

# xg = XGBRegressor(n_jobs = -1)

# params = {

#         'max_depth' : [10,20],

#         'learning_rate' : [0.1,0.2],

#         'n_estimators' : [100,200],

#         "subsample" : [0.5, 0.8]

        

#         }



# grid = GridSearchCV(estimator = xg,param_grid=params,cv = 5,n_jobs = -1)

# grid.fit(X_train,y_train)

# grid.best_params_
xg  = XGBRegressor(max_depth = 20,subsample=0.8).fit(X_train,y_train)

predic = xg.predict(X_test)

print('r2_score: ',r2_score(y_test,predic))

print('Accuracy: ',dt.score(X_test,predic))

print('RMSE: ',np.sqrt(mean_squared_error(y_test,predic)))

print('adj r sq.: ',1 - (1-r2_score(y_test,predic))*(len(y_test)-1)/(len(y_test) - X_test.shape[1]-1) )

xg.feature_importances_
feat_importances = pd.Series(xg.feature_importances_, index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')
X.head()
X.shape
from sklearn.preprocessing import StandardScaler
X_std = pd.DataFrame(StandardScaler().fit_transform(X),columns = X.columns)
cov_matrix = np.cov(X_std.T)

print(']n Covariance Matrix \n%s',cov_matrix)
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

print('Eigen Values \n%s',eig_vals)

print('Eigen Vectors \n%s',eig_vecs)
eigen_pairs = [(np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

tot  = sum(eig_vals)

var_exp = [(i/tot)*100 for i in sorted(eig_vals,reverse = True)]

cum_var_exp = np.cumsum(var_exp)

print('Cumulative VAriance Explained',cum_var_exp)
from sklearn.preprocessing import StandardScaler

scX = StandardScaler() 

X_train = scX.fit_transform(X_train) 

X_test = scX.fit_transform(X_test)
from sklearn.decomposition import PCA

pca = PCA(n_components = None) 

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)

explainedvariance = pca.explained_variance_ratio_
# rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,

#                       max_features='sqrt', max_leaf_nodes=None,

#                       min_impurity_decrease=0.0, min_impurity_split=None,

#                       min_samples_leaf=1, min_samples_split=2,

#                       min_weight_fraction_leaf=0.0, n_estimators=700, n_jobs=-1,

#                       oob_score=True, random_state=42, verbose=0,

#                       warm_start=False)

from sklearn.ensemble import AdaBoostRegressor

import xgboost as xgb

rf = xgb.XGBRegressor(max_depth=3)

predicts = rf.fit(X_train,y_train).predict(X_test)

np.sqrt(mean_squared_error(y_test,predicts))
# dftn = scX.fit_transform(dftn) 

# dftn = pca.fit_transform(dftn)
# predictrf = rf.predict(dftn)
# predictrfanti = np.exp(predictrf)-1
xg  = XGBRegressor(max_depth = 20,subsample=0.8).fit(X_train,y_train)

predic = xg.predict(X_test)

print('r2_score: ',r2_score(y_test,predic))

print('Accuracy: ',dt.score(X_test,predic))

print('RMSE: ',np.sqrt(mean_squared_error(y_test,predic)))

print('adj r sq.: ',1 - (1-r2_score(y_test,predic))*(len(y_test)-1)/(len(y_test) - X_test.shape[1]-1) )

t = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

submission_predicted = pd.DataFrame({'Id' : t['Id'],'SalePrice':predantilog })

submission_predicted.head()
submission_predicted.to_csv('submission.csv',index = False)