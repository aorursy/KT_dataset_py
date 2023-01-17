import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set()

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

from scipy import stats

from scipy.stats import norm, skew
train_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train_df.head()
train_df.columns.shape
train_df.columns
target=pd.DataFrame(train_df.iloc[:,-1],columns=['SalePrice'])

target.head()
train_df.drop('Id',axis=1,inplace=True)
train_df.isna().any()
missing=pd.DataFrame(train_df.isna().sum().sort_values(ascending=False)[0:19],columns=['Missing values'])

missing.reset_index(inplace=True)

missing.rename(columns={'index':'Feature name'},inplace=True)

missing
plt.figure(figsize=(10,8))

sns.barplot('Feature name','Missing values',data=missing)

plt.xticks(rotation=45)
plt.figure(figsize=(10,8))

sns.distplot(target['SalePrice'],fit=norm);

(mu, dev) = norm.fit(target['SalePrice'])

plt.xticks(rotation=45)

mu=np.round(target['SalePrice'].mean(),2)

dev=np.round(target['SalePrice'].std(),2)

plt.title('Target variable distribution')

plt.ylabel('Frequency')

plt.legend(['Normal dist. ($\mu=$ {} and $\sigma=$ {} )'.format(mu, dev)])
fig = plt.figure()

res = stats.probplot(target['SalePrice'], plot=plt)

plt.show()
from sklearn.preprocessing import power_transform
target['SalePrice'].describe()
target['Box-cox']=power_transform(target['SalePrice'].values.reshape(-1,1),method='box-cox',standardize=False)
sns.distplot(target['Box-cox'],fit=norm,color='indianred')

(mu, dev) = norm.fit(target['Box-cox'])

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, dev)])

plt.ylabel('Frequency')
fig = plt.figure()

res = stats.probplot(target['Box-cox'], plot=plt)

plt.show()
target['Log prices']=np.log(target['SalePrice'])
sns.distplot(target['Log prices'],fit=norm,color='green')

(mu, dev) = norm.fit(target['Log prices'])

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, dev)])

plt.ylabel('Frequency')
fig = plt.figure()

res = stats.probplot(target['Log prices'], plot=plt)

plt.show()
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

target['Min-max']=scaler.fit_transform(target['SalePrice'].values.reshape(-1,1))
sns.distplot(target['Min-max'],fit=norm)
fig = plt.figure()

res = stats.probplot(target['Min-max'], plot=plt)

plt.show()
from sklearn.preprocessing import RobustScaler
rob=RobustScaler()

target['Robust scaler']=rob.fit_transform(target['SalePrice'].values.reshape(-1,1))
sns.distplot(target['Robust scaler'],fit=norm)
fig = plt.figure()

res = stats.probplot(target['Robust scaler'], plot=plt)

plt.show()
plt.figure(figsize=(10,8))

sns.regplot(train_df['GrLivArea'],target['SalePrice'],line_kws={"color": "red"})

plt.xlabel('Living area')

plt.ylabel('Sale price')

plt.title('Sale price vs Living Area')
sns.boxplot(target['SalePrice'],orient='v')
target['SalePrice'].median()
train_df = train_df.drop(train_df[(train_df['GrLivArea']>4500) & (train_df['SalePrice']<350000)].index)
plt.figure(figsize=(10,8))

sns.regplot(train_df['GrLivArea'],train_df['SalePrice'],line_kws={"color": "red"})

plt.xlabel('Living area')

plt.ylabel('Sale price')

plt.title('Sale price vs Living Area without outliers')
plt.figure(figsize=(10,8))

sns.regplot(train_df['TotalBsmtSF'],train_df['SalePrice'],line_kws={'color':'green'})
plt.figure(figsize=(10,8))

sns.boxplot('YrSold','SalePrice',data=train_df)
plt.figure(figsize=(10,8))

sns.boxplot('OverallQual','SalePrice',data=train_df)

plt.title('Overall quality Vs Sale prices')
plt.figure(figsize=(10,8))

sns.regplot('GarageArea','SalePrice',data=train_df,line_kws={'color':'red'})

plt.title('Gara area vs Sale price')
train_df=train_df.drop(train_df[(train_df['GarageArea']>1200)&(train_df['SalePrice']<300000)].index)
train_df=train_df.drop(train_df[(train_df['GarageArea']>600)&(train_df['GarageArea']<1000)&(train_df['SalePrice']>550000)].index)
plt.figure(figsize=(10,8))

sns.regplot('GarageArea','SalePrice',data=train_df,line_kws={'color':'red'})

plt.title('Gara area vs Sale price')

plt.ylim(0,700000)
plt.figure(figsize=(10,8))

sns.boxplot('TotRmsAbvGrd','SalePrice',data=train_df)

plt.title('Total rooms above ground Vs Sale Price')
plt.figure(figsize=(20,8))

sns.boxplot('YearBuilt','SalePrice',data=train_df)

plt.xticks(rotation=90)
corr=train_df.corr()

plt.figure(figsize=(20,20))

sns.heatmap(corr,annot=True)
k=10

cols=corr.nlargest(k,'SalePrice')['SalePrice'].index

cols
corr_highest=train_df[cols].corr()

plt.figure(figsize=(10,8))

sns.heatmap(corr_highest,annot=True,fmt='g')
plt.figure(figsize=(10,8))

sns.boxplot(train_df['FullBath'],train_df['SalePrice'])

plt.title('SalePrice Vs Full bath')
plt.figure(figsize=(20,8))

sns.boxplot(train_df['YearRemodAdd'],train_df['SalePrice'])

plt.title('Year remodelled Vs Sale Price')

plt.xticks(rotation=45)
train_df.head()
train_df.isna().any()
train_df['LotFrontage'].describe()
sns.boxplot(train_df['LotFrontage'],orient='v')
train_df['LotFrontage'].fillna(train_df['LotFrontage'].median(),inplace=True)
train_df['Alley'].isna().value_counts()
train_df['Alley'].fillna('None',inplace=True)
train_df['MasVnrType'].isna().value_counts()
train_df['MasVnrType'].value_counts()
train_df['MasVnrType'].fillna(train_df['MasVnrType'].mode()[0],inplace=True)
train_df['MasVnrArea'].isna().value_counts()
train_df['MasVnrArea'].fillna(train_df['MasVnrArea'].median(),inplace=True)
basement_features=['BsmtQual' ,'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']



for feature in basement_features:

    train_df[feature]=train_df[feature].fillna('None')
train_df['BsmtQual'].isna().value_counts()
train_df['BsmtQual'].value_counts()
train_df['Electrical'].isna().value_counts()
train_df['Electrical'].fillna(train_df['Electrical'].mode()[0],inplace=True)
misc_features=['FireplaceQu', 'GarageType', 'GarageYrBlt','GarageFinish' ,

               'GarageQual','GarageCond' ,'PoolQC','Fence','MiscFeature' ]
for misc in misc_features:

    train_df[misc].fillna('None',inplace=True)
train_df.isna().any()
from sklearn.preprocessing import OrdinalEncoder

oe=OrdinalEncoder()
cat_features=['FireplaceQu', 'BsmtQual', 'BsmtCond','GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold']
for cat in cat_features:

    train_df[cat]=oe.fit_transform(train_df[cat].values.reshape(-1,1))

    
train_df.head()
train_df.drop('GarageYrBlt',axis=1,inplace=True)
train_df=pd.get_dummies(train_df)
train_df.dtypes
target_df=pd.DataFrame(train_df['SalePrice'],columns=['SalePrice'])

target_df.head()
train_df.drop('SalePrice',axis=1,inplace=True)
train_df.head()
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

X_scaled_input=scaler.fit_transform(train_df)
target_df['LogSalePrice']=np.log(target_df['SalePrice'])

target_df.head()
from sklearn.linear_model import LinearRegression,Lasso,Ridge

from sklearn.metrics.regression import r2_score,explained_variance_score

from sklearn.model_selection import train_test_split
reg_lin=LinearRegression()
X=train_df

y=target_df['SalePrice'].values.astype(float)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
reg_lin.fit(X_train,y_train)
y_pred=reg_lin.predict(X_test)
reg_lin.score(X_train,y_train)
reg_lin.score(X_test,y_test)
lin_pred_df=pd.DataFrame(columns=['Actual values','Predicted values'])
lin_pred_df['Actual values']=y_test

lin_pred_df['Predicted values']=y_pred

lin_pred_df['Absolute difference']=abs(lin_pred_df['Actual values']-lin_pred_df['Predicted values'])

lin_pred_df['Residual']=lin_pred_df['Actual values']-lin_pred_df['Predicted values']

lin_pred_df.head()
lin_pred_df['Residual'].describe()
plt.scatter(lin_pred_df['Actual values'],lin_pred_df['Predicted values'])
sns.distplot(lin_pred_df['Residual'])
X=X_scaled_input

y=target_df['LogSalePrice'].values.astype(float)

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

reg_las=Lasso(alpha =0.0005, random_state=1,normalize=True)
reg_las.fit(X_train,y_train)
y_pred=reg_las.predict(X_test)
reg_las.score(X_test,y_test)
reg_las.score(X_train,y_train)
las_pred_df=pd.DataFrame(columns=['Actual values','Predicted values','Absolute difference'])

las_pred_df['Actual values']=np.exp(y_test)

las_pred_df['Predicted values']=np.exp(y_pred)

las_pred_df['Absolute difference']=abs(las_pred_df['Actual values']-las_pred_df['Predicted values'])

las_pred_df['Residual']=las_pred_df['Actual values']-las_pred_df['Predicted values']

las_pred_df['Difference %']=100*las_pred_df['Residual']/las_pred_df['Actual values']

las_pred_df
las_pred_df['Absolute difference'].describe()


sns.distplot(las_pred_df['Residual'])

plt.title('Residual PDF',size=20)
sns.regplot(las_pred_df['Actual values'],las_pred_df['Predicted values'],line_kws={'color':'red'})
bias=np.exp(reg_las.intercept_)

bias
reg_rid=Ridge()
reg_rid.fit(X_train,y_train)
y_pred=reg_rid.predict(X_test)
reg_rid.score(X_train,y_train)
reg_rid.score(X_test,y_test)
r2_score(y_pred,y_test)
explained_variance_score(y_pred,y_test)
rid_df=pd.DataFrame(columns=['Actual values','Predicted values','Absolute difference'])

rid_df['Actual values']=np.exp(y_test)

rid_df['Predicted values']=np.exp(y_pred)

rid_df['Absolute difference']=abs(rid_df['Actual values']-rid_df['Predicted values'])

rid_df['Residual']=rid_df['Actual values']-rid_df['Predicted values']

rid_df['Difference %']=100*rid_df['Residual']/rid_df['Actual values']

rid_df.head()
sns.distplot(rid_df['Residual'])

plt.title('Residual PDF',size=20)
sns.regplot(rid_df['Actual values'],rid_df['Predicted values'],line_kws={'color':'red'})
from sklearn.ensemble import RandomForestRegressor
X=train_df

y=target_df['SalePrice']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True,random_state=0)
rfr=RandomForestRegressor()
rfr.fit(X_train,y_train)
y_pred=rfr.predict(X_test)
rfr.score(X_train,y_train)
rfr.score(X_test,y_test)
rfr_df=pd.DataFrame(columns=['Actual values','Predicted values','Absolute difference'])

rfr_df['Actual values']=y_test

rfr_df['Predicted values']=y_pred

rfr_df['Absolute difference']=abs(rfr_df['Actual values']-rfr_df['Predicted values'])

rfr_df['Residual']=rfr_df['Actual values']-rfr_df['Predicted values']

rfr_df['Difference %']=100*rfr_df['Residual']/rfr_df['Actual values']

rfr_df.head()
sns.distplot(rfr_df['Residual'])

plt.title('Residual PDF',size=20)
sns.regplot(rfr_df['Actual values'],rfr_df['Predicted values'],line_kws={'color':'green'})
from sklearn.ensemble import GradientBoostingRegressor
gbdt=GradientBoostingRegressor()
gbdt.fit(X_train,y_train)
y_pred=gbdt.predict(X_test)
gbdt.score(X_train,y_train)
gbdt.score(X_test,y_test)
gbdt_df=pd.DataFrame(columns=['Actual values','Predicted values','Absolute difference'])

gbdt_df['Actual values']=y_test

gbdt_df['Predicted values']=y_pred

gbdt_df['Absolute difference']=abs(gbdt_df['Actual values']-gbdt_df['Predicted values'])

gbdt_df['Residual']=gbdt_df['Actual values']-gbdt_df['Predicted values']

gbdt_df['Difference %']=100*gbdt_df['Residual']/gbdt_df['Actual values']

gbdt_df.head()
sns.distplot(gbdt_df['Residual'])
sns.regplot(gbdt_df['Actual values'],gbdt_df['Predicted values'],line_kws={'color':'red'})
import xgboost as xgb
xgb_reg=xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)
xgb_reg.fit(X_train,y_train)
xgb_reg.score(X_train,y_train)
xgb_reg.score(X_test,y_test)
y_pred=xgb_reg.predict(X_test)
xgb_df=pd.DataFrame(columns=['Actual values','Predicted values','Absolute difference'])

xgb_df['Actual values']=y_test

xgb_df['Predicted values']=y_pred

xgb_df['Absolute difference']=abs(xgb_df['Actual values']-xgb_df['Predicted values'])

xgb_df['Residual']=xgb_df['Actual values']-xgb_df['Predicted values']

xgb_df['Difference %']=100*xgb_df['Residual']/xgb_df['Actual values']

xgb_df.head()
sns.distplot(xgb_df['Residual'])
sns.regplot(xgb_df['Actual values'],xgb_df['Predicted values'],line_kws={'color':'green'})

plt.title('Prediction chart',size=20)
from lightgbm import LGBMRegressor
lgb=LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
X=train_df

y=target_df['SalePrice'].values.astype(float)

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
lgb.fit(X_train,y_train)
lgb.score(X_train,y_train)
lgb.score(X_test,y_test)
y_pred=lgb.predict(X_test)
lgb_df=pd.DataFrame(columns=['Actual values','Predicted values','Absolute difference'])

lgb_df['Actual values']=y_test

lgb_df['Predicted values']=y_pred

lgb_df['Absolute difference']=abs(lgb_df['Actual values']-lgb_df['Predicted values'])

lgb_df['Residual']=lgb_df['Actual values']-lgb_df['Predicted values']

lgb_df['Difference %']=100*lgb_df['Residual']/lgb_df['Actual values']

lgb_df.head()
sns.distplot(lgb_df['Residual'])

plt.title('Residual PDF',size=20)
sns.regplot(lgb_df['Actual values'],lgb_df['Predicted values'],line_kws={'color':'red'})

plt.title('Prediction chart',size=20)
test_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test_df.head()
train_df.head()
test_id=pd.DataFrame(test_df.iloc[:,0],columns=['Id'])

test_id.head()
test_df.isna().any()
missing_test=pd.DataFrame(test_df.isna().sum().sort_values(ascending=False)[0:33],columns=['Missing values'])

missing_test.reset_index(inplace=True)

missing_test.rename(columns={'index':'Feature name'},inplace=True)

missing_test
plt.figure(figsize=(10,8))

sns.barplot('Feature name','Missing values',data=missing_test)

plt.xticks(rotation=90)
test_df['LotFrontage'].fillna(test_df['LotFrontage'].median(),inplace=True)

test_df['Alley'].fillna('None',inplace=True)

test_df['MasVnrType'].fillna(test_df['MasVnrType'].mode()[0],inplace=True)

test_df['MasVnrArea'].fillna(test_df['MasVnrArea'].median(),inplace=True)

basement_features=['BsmtQual' ,'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']



for feature in basement_features:

    test_df[feature]=test_df[feature].fillna('None')

    

test_df['Electrical'].fillna(test_df['Electrical'].mode()[0],inplace=True)





misc_features=['FireplaceQu', 'GarageType', 'GarageYrBlt','GarageFinish' ,

               'GarageQual','GarageCond' ,'PoolQC','Fence','MiscFeature' ]



for misc in misc_features:

    test_df[misc].fillna('None',inplace=True)



test_df['MSZoning'].fillna(test_df['MSZoning'].mode()[0],inplace=True)

test_df['Utilities'].fillna(test_df['Utilities'].mode()[0],inplace=True)

test_df['Exterior1st'].fillna(test_df['Exterior1st'].mode()[0],inplace=True)

test_df['Exterior2nd'].fillna(test_df['Exterior2nd'].mode()[0],inplace=True)

test_df['BsmtFinSF1'].fillna(test_df['BsmtFinSF1'].median(),inplace=True)

test_df['BsmtFinSF2'].fillna(test_df['BsmtFinSF2'].median(),inplace=True)

test_df['BsmtUnfSF'].fillna(test_df['BsmtUnfSF'].median(),inplace=True)

test_df['TotalBsmtSF'].fillna(test_df['TotalBsmtSF'].median(),inplace=True)

test_df['KitchenQual'].fillna(test_df['KitchenQual'].mode()[0],inplace=True)





baths=['BsmtFullBath','BsmtHalfBath']

for types in baths:

    test_df[types].fillna(test_df[types].mode()[0],inplace=True)

    

test_df['Functional'].fillna(test_df['Functional'].mode()[0],inplace=True)

test_df['GarageCars'].fillna(test_df['GarageCars'].mode()[0],inplace=True)

test_df['GarageArea'].fillna(test_df['GarageArea'].median(),inplace=True)

test_df['SaleType'].fillna(test_df['SaleType'].mode()[0],inplace=True)
test_df.isna().any()
for cat in cat_features:

    test_df[cat]=oe.fit_transform(test_df[cat].values.reshape(-1,1))

    
test_df.drop('GarageYrBlt',axis=1,inplace=True)
test_df.drop('Id',axis=1,inplace=True)
test_df=pd.get_dummies(test_df)
test_df.dtypes
missing_cols=['Condition2_RRAe', 'Utilities_NoSeWa', 'RoofMatl_Metal', 

              'Condition2_RRAn', 'Exterior2nd_Other', 'MiscFeature_TenC', 

              'RoofMatl_Roll', 'Electrical_Mix', 'RoofMatl_Membran', 

              'Exterior1st_ImStucc', 'Condition2_RRNn', 'Heating_Floor', 

              'Heating_OthW','HouseStyle_2.5Fin', 'Exterior1st_Stone']
for cols in missing_cols:

    test_df[cols]=0
test_df=test_df[train_df.columns]
train_df.head()
test_df.head()
X_test=test_df

X_scaled_test=scaler.fit_transform(X_test)
y_pred=reg_lin.predict(X_test)
linear_reg_final=pd.DataFrame(columns=['Id','SalePrice'])

linear_reg_final['Id']=test_id['Id']

linear_reg_final['SalePrice']=y_pred

linear_reg_final.head()
y_pred=reg_las.predict(X_scaled_test)
lasso_final=pd.DataFrame(columns=['Id','SalePrice'])

lasso_final['Id']=test_id['Id']

lasso_final['SalePrice']=np.exp(y_pred)

lasso_final.head()
y_pred=reg_rid.predict(X_scaled_test)
ridge_final=pd.DataFrame(columns=['Id','SalePrice'])

ridge_final['Id']=test_id['Id']

ridge_final['SalePrice']=np.exp(y_pred)

ridge_final.head()
y_pred=rfr.predict(X_test)
rf_final=pd.DataFrame(columns=['Id','SalePrice'])

rf_final['Id']=test_id['Id']

rf_final['SalePrice']=y_pred

rf_final.head()
y_pred=xgb_reg.predict(X_test)
xgb_final=pd.DataFrame(columns=['Id','SalePrice'])

xgb_final['Id']=test_id['Id']

xgb_final['SalePrice']=y_pred

xgb_final.head()
y_pred=gbdt.predict(X_test)
gbdt_final=pd.DataFrame(columns=['Id','SalePrice'])

gbdt_final['Id']=test_id['Id']

gbdt_final['SalePrice']=y_pred

gbdt_final.head()
y_pred=lgb.predict(X_test)
lgb_final=pd.DataFrame(columns=['Id','SalePrice'])

lgb_final['Id']=test_id['Id']

lgb_final['SalePrice']=y_pred

lgb_final.head()
lgb_final.to_csv('LGBM_Regression.csv',index=False)