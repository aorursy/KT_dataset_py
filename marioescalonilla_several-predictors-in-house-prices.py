import numpy as np

from numpy import *

import pandas as pd   

import seaborn as sns #Nice way to show your images

import matplotlib.pyplot as plt

from scipy import stats

from scipy.stats import norm, skew #Normalize the  skewness

from sklearn.preprocessing import RobustScaler  #Scaling before pipeline

from scipy.stats import pearsonr

from scipy.stats import norm # for skewness ( not used)

from scipy.special import boxcox1p # for negative skewness

from sklearn.metrics import mean_squared_log_error #metric to evaluate predictions

from sklearn.model_selection import KFold, cross_val_score





#Regressors

from sklearn.pipeline import make_pipeline # transforming steps pipeline

from sklearn.svm import SVR

import xgboost

import lightgbm as lgbm

from xgboost import plot_importance

from sklearn.linear_model import ElasticNet,ElasticNetCV

from sklearn.linear_model import Lasso,LassoCV

from sklearn.linear_model import Ridge, RidgeCV

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import StackingRegressor



from statsmodels.stats.outliers_influence import variance_inflation_factor # For calculating VIF
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test_ID = df_test['Id']
df_train.head(10)
df_train.info(5)

corr_matrix= df_train.corr()

f, ax = plt.subplots(figsize=(12,10))

cmap = sns.diverging_palette(255, 5,as_cmap=True)

sns.heatmap(corr_matrix,cmap=cmap,vmax=0.8,center=0, linewidths=.9)
strong_rel = ['TotalBsmtSF','GarageCars','GarageArea','GrLivArea','OverallQual','1stFlrSF']

for var in strong_rel:

    corr, _ = pearsonr(df_train[var], df_train['SalePrice'])

    print(var + "--> type:"+ str(df_train[var].dtypes)  + "; Pearson_corr_value:" + str(round(corr, 2)) + "; Unique values:" + str(len(df_train[var].unique())))
corr, _ = pearsonr(df_train['GarageCars'], df_train['GarageArea'])

print('Pearson coefficient between GarageCars and GarageArea :' +str(corr))
for var in strong_rel:

    print(var + " has "+str(df_train[var].isna().sum()) + " missing values")
for var in strong_rel:

    print(var + " has "+str(df_test[var].isna().sum()) + " missing values")
plt.figure(figsize=(15,7))



ax=sns.distplot(df_train['SalePrice'] , fit=norm,color="y")

ax.axes.set_title("SalePrice Distribution ",fontsize=20) 





plt.figure(figsize=(15,7))

res = stats.probplot(df_train['SalePrice'], plot=plt)
df_train["SalePrice"] = np.log1p(df_train["SalePrice"])

plt.figure(figsize=(15,7))



ax=sns.distplot(df_train['SalePrice'] , fit=norm,color="y");

ax.axes.set_title("SalePrice Distribution ",fontsize=20) 





plt.figure(figsize=(15,7))

res = stats.probplot(df_train['SalePrice'], plot=plt)
skewValue = df_train["SalePrice"].skew()

skewValue
m=sns.catplot(x="GarageCars", y="SalePrice", kind="box", data=df_train)
len(df_train.loc[df_train['GarageCars']==4])
garage_3cars =df_train.loc[(df_train['GarageCars']==3) & (df_train['SalePrice']>12.3),['OverallQual','GarageCars','GarageArea','GrLivArea','SalePrice']].copy()

garage_3cars
garage_4cars=df_train.loc[(df_train['GarageCars']==4) & (df_train['SalePrice']<12.4),['OverallQual','GarageCars','GarageArea','GrLivArea','SalePrice']].copy()

garage_4cars
print("Average Quality of a  garage for 4 cars: "+ str(garage_4cars['OverallQual'].mean()))

print("Average Quality of a  garage for 3 cars: "+ str(round(garage_3cars['OverallQual'].mean(),2)))
ax = sns.catplot(x="OverallQual", y="SalePrice", kind="box", data=df_train)

ax.set(xlabel='Overall Quality', ylabel='Sale Price')
outliers=  df_train.loc[(df_train['OverallQual'] <5 )&(df_train['SalePrice']>12.3),['OverallQual','SalePrice']]

outliers
df_train.drop(outliers.index,inplace = True)
pal = sns.color_palette("husl", 8)

g = sns.FacetGrid(df_train, hue="GarageCars", palette=pal, height=5)

g.map(plt.scatter, "GarageArea", "SalePrice", s=50, alpha=.7, linewidth=.5, edgecolor="white")

g.add_legend();
pal = sns.color_palette("husl", 8)

g = sns.FacetGrid(df_train, hue="GarageCars", palette=pal, height=5)

g.map(plt.scatter, "GrLivArea", "SalePrice", s=50, alpha=.7, linewidth=.5, edgecolor="white")

g.add_legend();
g = sns.FacetGrid(df_train, palette=pal, height=5)

g.map(sns.regplot,"GrLivArea", "SalePrice")

outliers=  df_train.loc[(df_train['GrLivArea'] > 4500 )&(df_train['SalePrice']<13)&(df_train['SalePrice']>11.5),['GrLivArea','SalePrice'] ]

outliers
df_train.drop(outliers.index, inplace = True)

df_train.reset_index(drop = True)
g = sns.FacetGrid(df_train, height=5, palette="GnBu_d")

g.map(sns.regplot,"GrLivArea", "SalePrice")

print( "Train shape before:" + str(df_train.shape))

print( "Test shape before:" + str(df_test.shape))



#We dont need the Id column 

df_train.drop("Id", axis = 1, inplace = True)

y_train = df_train['SalePrice']

df_train.drop(['SalePrice'], axis=1, inplace=True)

df_test.drop("Id", axis = 1, inplace = True)



print("-----------------------------------")

print( "Train shape after:" + str(df_train.shape))

print( "Test shape after:" + str(df_test.shape))



dim_train = df_train.shape[0]



df_all = pd.concat((df_train,df_test)).reset_index(drop=True)
df_all['SaleType'].isna().sum()
df_all['SaleType'] = df_all['SaleType'].fillna(df_all['SaleType'].mode()[0])
no_rows = len(df_all.index)

nameVar = []

missing_val = []

for var in df_all.columns:

    missing_per = round(((df_all[var].isna().sum())/no_rows)*100,2)

    if (missing_per) > 0:

        nameVar.append(var)

        missing_val.append(missing_per)



df_missVar= pd.DataFrame({'Variables':nameVar,'Percentage Missing':missing_val})

df_missVar = df_missVar.sort_values(by='Percentage Missing', ascending = False)



if(len(df_missVar.values) >= 1):

    plt.figure(figsize=(20,10))

    b = sns.barplot(x = df_missVar['Variables'],

                    y=df_missVar['Percentage Missing'])

    b.axes.set_title("Percentage of Missing values ",fontsize=20) 

    b.set_xlabel("Features",fontsize=10)

    b.set_ylabel("% Missing values",fontsize=10)

    b.tick_params(axis = 'x',labelsize=10,rotation=90)

    b.tick_params(axis = 'y',labelsize=10)

    

    missing_values=df_missVar['Percentage Missing'].values

    for i,index in enumerate(missing_values):

        b.text(i,index, str(round(index,4)), color='black', ha="center",fontsize=10)
df_all['PoolQC'] = df_all['PoolQC'].fillna('None')

df_all['Alley'] = df_all['Alley'].fillna('None')

df_all['Fence'] = df_all['Fence'].fillna('None')

df_all['MiscFeature'] = df_all['MiscFeature'].fillna('None')
df_all['LotFrontage'] = df_all.groupby(

    'Neighborhood')['LotFrontage'].transform(

    lambda x: x.fillna(x.median()))
df_all['MasVnrType'] = df_all['MasVnrType'].fillna('None')
print("There are : " +str(df_all['GarageYrBlt'].notna().sum()) + ' GarageYrBlt missing values')
print( str(round((df_all['GarageYrBlt'].isna().sum()/len(df_all['GarageYrBlt'])*100), 2)) + "% of missing values")
df_all.loc[df_all['GarageYrBlt'].isna(),['GarageYrBlt','GarageType','GarageCond','GarageFinish','GarageQual']]
df_all['GarageYrBlt']=  df_all['GarageYrBlt'].fillna(0)
df_all['GarageType'] = df_all['GarageType'].fillna('None')

df_all['GarageQual'] = df_all['GarageQual'].fillna('None')

df_all['GarageCond'] = df_all['GarageCond'] .fillna('None')

df_all['GarageFinish'] = df_all['GarageFinish'].fillna('None')
df_all['MasVnrArea'].isna().sum()
df_all.loc[df_all['MasVnrArea'].isna(),['MasVnrArea','MasVnrType']]
df_all['MasVnrArea']=df_all['MasVnrArea'].fillna(0)
df_all['BsmtFinType1'] = df_all['BsmtFinType1'].fillna('None')

df_all['BsmtFinType2'] = df_all['BsmtFinType2'].fillna('None')

df_all['BsmtQual']     = df_all['BsmtQual'].fillna('None')

df_all['BsmtCond']     = df_all['BsmtCond'].fillna('None')

df_all['BsmtExposure'] = df_all['BsmtExposure'].fillna('None')
df_all['Electrical'].isna().sum()
sns.countplot(df_all['Electrical'])
df_all['Electrical'] = df_all['Electrical'].fillna('SBrkr')
df_all['FireplaceQu'] = df_all['FireplaceQu'].fillna('None')
df_all['MSZoning'].isna().sum()
sns.countplot(df_all['MSZoning'])
df_all['MSZoning'] = df_all['MSZoning'].fillna(df_all['MSZoning'].mode()[0])
df_all['Functional'].isna().sum()
sns.countplot(df_all['Functional'])
df_all['Functional'] = df_all['Functional'].fillna('Typ')
sns.countplot(df_all['Utilities'])
df_all['Utilities'].describe()
len(df_all.loc[df_all['Utilities'] == 'NoSeWa'])
df_all.drop('Utilities',axis=1,inplace=True) 
sns.countplot(df_all['SaleType'])
df_all['SaleType'] = df_all['SaleType'].fillna(df_all['SaleType'].mode()[0])
sns.countplot(df_test['KitchenQual'])
df_all['KitchenQual'].isna().sum()
df_all['KitchenQual'] = df_all['KitchenQual'].fillna(df_all['KitchenQual'].mode()[0])
df_all['GarageCars'] = df_all['GarageCars'].fillna(0)

df_all['GarageArea'] = df_all['GarageArea'].fillna(0)
bsmt_vars = ['BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF', 'BsmtFullBath','BsmtHalfBath','TotalBsmtSF']

for var in bsmt_vars:

    df_all[var]  = df_all[var].fillna(0)
df_all['Exterior2nd'] = df_all['Exterior2nd'].fillna(df_all['Exterior2nd'].mode()[0])

df_all['Exterior1st'] = df_all['Exterior1st'].fillna(df_all['Exterior1st'].mode()[0])

some_variables = ['MSSubClass','MoSold']

for var in some_variables:

    print(var + ' has a type ' +str(df_all[var].dtypes))
for var in some_variables:

    df_all[var]  = df_all[var].apply(str)
some_var =['GarageYrBlt','GarageCars','BsmtHalfBath','BsmtFullBath']      

for var in some_var:

    print(var + ' has a type ' +str(df_all[var].dtypes))
for var in some_variables:

    df_all[var]  = df_all[var].apply('int64')
corr= pearsonr(df_train['GarageArea'], df_train['GarageCars'])

corr
f ,(ax1,ax2)= plt.subplots(1,2,figsize=(10,5))

sns.barplot(x=df_all['GarageCars'], y= df_all['GarageArea'],ax=ax1)

sns.regplot(x="GarageCars", y="GarageArea", data=df_all, ax=ax2);
df_all.drop('GarageArea',axis=1,inplace=True)
continuous_var =['LotFrontage'   ,'LotArea'    ,'OverallQual'    ,

           'OverallCond'      ,

           'MasVnrArea'   ,'BsmtFinSF1'   ,'BsmtFinSF2'   ,

           'BsmtUnfSF'   ,'TotalBsmtSF'   ,'1stFlrSF'   ,

           '2ndFlrSF'   ,'LowQualFinSF'   ,'GrLivArea'   ,

           'BsmtFullBath','BsmtHalfBath'    ,'FullBath'    ,

           'HalfBath'    ,'BedroomAbvGr'    ,'KitchenAbvGr'    ,

           'TotRmsAbvGrd'    ,'Fireplaces'      ,'GarageYrBlt' ,

           'GarageCars'      ,'WoodDeckSF' ,

           'OpenPorchSF'     ,'EnclosedPorch'   ,'3SsnPorch',

           'ScreenPorch'    ,'PoolArea'        ,'MiscVal'  ,'MoSold'         ,'YrSold']          
df_continuous = df_all.copy()

df_continuous = df_continuous[continuous_var]



corr_matrix= df_continuous.corr()

f, ax = plt.subplots(figsize=(12,10))

cmap = sns.diverging_palette(255, 5,as_cmap=True)

sns.heatmap(corr_matrix,cmap=cmap,vmax=0.8,center=0, linewidths=.9)
corr = pearsonr(df_all['1stFlrSF'], df_all['TotalBsmtSF'])

corr
sns.jointplot(x="1stFlrSF", y="TotalBsmtSF", data=df_all, kind="reg");
corr = pearsonr(df_train['1stFlrSF'], y_train)

corr
corr = pearsonr(df_train['TotalBsmtSF'], y_train)

corr
#df_all['1st_BsmtSF'] = df_all['1stFlrSF'] + df_all['TotalBsmtSF']

#df_all.drop('1stFlrSF',axis=1,inplace=True)
numeric_types = ['int16','int32','int64','float16','float32','float64']

numeric_skew = pd.DataFrame()

numerical_columns  = ['Name','Skew']

data = []

skewed_names = []

for col in df_all.columns:

    if df_all[col].dtypes in numeric_types:

        if (df_all[col].skew()) > 0.5:

            

            skewed_names.append(col)

            values = [col,df_all[col].skew()]

            temp_dic = zip(numerical_columns,values)

            data.append(dict(temp_dic))

numeric_skew = numeric_skew.append(data,ignore_index=False)

numeric_skew = numeric_skew.sort_values(by='Skew',ascending=False)

numeric_skew



plt.figure(figsize=(30,10))

b = sns.barplot(x = numeric_skew['Name'],

                y=numeric_skew['Skew'])

b.axes.set_title("Skewness of Numeric values",fontsize=20)

b.set_xlabel("Features",fontsize=15)

b.set_ylabel("Skew value",fontsize=15)

b.tick_params(axis = 'x',labelsize=15,rotation=90)

b.tick_params(axis = 'y',labelsize=15)





numeric_skewed = df_all[skewed_names].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = numeric_skewed[numeric_skewed > 0.5]
from scipy.stats import boxcox_normmax

skew_index = high_skew.index



for i in skew_index:

    df_all[i] =  boxcox1p(df_all[i],boxcox_normmax(df_all[i] +1 ))
normalized_values = pd.DataFrame({'Skew':df_all[skew_index].skew()}).sort_values(by='Skew', ascending = False)
f ,(ax1,ax2)= plt.subplots(2,1,figsize=(30,20) )

plt.subplots_adjust( hspace = 0.5   )

c = sns.barplot(x = numeric_skew['Name'],

                y=numeric_skew['Skew'],palette="ch:.25", ax = ax1)

c.axes.set_title("Previous Skewness",fontsize=20)

c.set_xlabel("Features",fontsize=15)

c.set_ylabel("Skew value",fontsize=15)

c.tick_params(axis = 'x',labelsize=15,rotation=90)

c.tick_params(axis = 'y',labelsize=15)



b = sns.barplot(y = normalized_values.Skew , x = normalized_values.index,palette="ch:.25" , ax = ax2)

b.axes.set_title("Actual Skewness",fontsize=20)

b.set_xlabel("Features",fontsize=15)

b.set_ylabel("Skew value",fontsize=15)

b.tick_params(axis = 'x',labelsize=15,rotation=90)

b.tick_params(axis = 'y',labelsize=15)
#sns.distplot(df_all['PoolArea'] , fit=norm,color="y");
df_all = pd.get_dummies(df_all).reset_index(drop=True)

X = df_all.copy()

X = X[:dim_train]

y = y_train.copy()

test = df_all.copy()

test  = test[dim_train:]
kf = KFold(n_splits=5, random_state=42, shuffle=True).get_n_splits(X.values)

def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
lr_rg = LinearRegression()

score_lr = round(rmse_cv(lr_rg).mean(),5)

score_lr_std = round(rmse_cv(lr_rg).std(),5)

print("Linear Regression :", str(score_lr)+ ', std : '+str(score_lr_std))
svr_reg = make_pipeline(RobustScaler(), SVR(

    C= 20, epsilon= 0.008, gamma=0.0003))

svr_score = round(rmse_cv(svr_reg).mean(),5)

svr_score_std = round(rmse_cv(svr_reg).std(),5)

print("SVR: "+ str(svr_score) + ', std : '+ str(svr_score_std))
alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3,4,5, 6, 10, 30, 60]

ridge_reg = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas, cv=kf))

ridge_score = round(rmse_cv(ridge_reg).mean(),5)

ridge_score_std = round(rmse_cv(ridge_reg).std(),5)

print("Ridge: "+ str(ridge_score) + ', std : '+ str(ridge_score_std))
lasso_reg = make_pipeline(RobustScaler(), LassoCV(alphas=np.arange(0.0001,0.1,0.0001),max_iter = 50000, cv=kf))

lasso_score = round(rmse_cv(lasso_reg).mean(),5)

lasso_score_std = round(rmse_cv(lasso_reg).std(),5)

print("Lasso: "+ str(lasso_score)+ ', std : '+str(lasso_score_std))
elnet_reg = make_pipeline(RobustScaler(),ElasticNetCV(

    l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8,0.85, 0.9, 0.95, 1],

    alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 

                     0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 

    max_iter = 50000, cv = kf))



score_elnet = round(rmse_cv(elnet_reg).mean(),5)

score_elnet_std = round(rmse_cv(elnet_reg).std(),5)

print("Elastic Net: "+ str(score_elnet)+ ', std : '+str(score_elnet_std))

# Choosing the right parameters took a looooong time with Grid method... Here is the one that best worked for me.

xgb_reg = xgboost.XGBRegressor(

        colsample_bytree=0.4580, gamma=0.05, reg_alpha=0.5, reg_lambda=0.8,

        subsample=0.55, silent=1,learning_rate=0.07, max_depth=3, 

        min_child_weight=1.6, n_estimators=3000,random_state =7,

        nthread = -1)



score_xgb = round(rmse_cv(xgb_reg).mean(),5)

score_xgb_std = round(rmse_cv(xgb_reg).std(),5)

print("XGBoost: "+ str(score_xgb)+ ', std : '+ str(score_xgb_std))
lgbm_reg = lgbm.LGBMRegressor(objective='regression',num_leaves=6,

            feature_fraction_seed=9, bagging_seed=9,learning_rate=0.05,

            n_estimators=790,bagging_fraction = 0.9,max_bin = 52,

            feature_fraction = 0.2340, bagging_freq = 5,min_data_in_leaf =7,

             min_sum_hessian_in_leaf = 12)

score_lgbm= round(rmse_cv(lgbm_reg).mean(),5)

score_lgbm_std= round(rmse_cv(lgbm_reg).std(),5)

print("LightGBM: "+ str(score_lgbm)+ ', std : '+str(score_lgbm_std))
gbr_reg = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

score_gbr= round(rmse_cv(gbr_reg).mean(),5)

score_gbr_std= round(rmse_cv(gbr_reg).std(),5)

print("GBR: "+ str(score_gbr)+ ', std : '+str(score_gbr_std))
estimators = [

     ('Ridge', ridge_reg), ('Lasso',lasso_reg),

     ('svr', svr_reg)

 ]

stk_reg = StackingRegressor(

     estimators=estimators,

     final_estimator=elnet_reg)



regressor_df = pd.DataFrame({

                    'Regressor' : ['Linear Regression','SVR','Ridge',

'Lasso','Elastic Net','XGBoost','LightGBM','Gradient Boost Regressor'],

                     'RMSLE' :[ score_lr, svr_score, ridge_score,

 lasso_score, score_elnet, score_xgb, score_lgbm,score_gbr],

                    'RMSLE deviation':[score_lr_std,svr_score_std,

ridge_score_std,lasso_score_std,score_elnet_std,

score_xgb_std,score_lgbm_std,score_gbr_std]} )
pal = sns.cubehelix_palette(8, start=.5, rot=-.75)

f ,(ax1,ax2)= plt.subplots(2,1,figsize=(30,20) )

plt.subplots_adjust( hspace = 0.9  )



regressor_df = regressor_df.sort_values(by='RMSLE deviation', ascending = False)



c = sns.barplot(x = regressor_df['Regressor'],

                y=regressor_df['RMSLE deviation'],palette=pal,ax=ax2)

c.axes.set_title(" Standard Deviation of the Models ",fontsize=20)

c.set_xlabel("Models",fontsize=17)

c.set_ylabel("RMSLE deviation",fontsize=17)

c.tick_params(axis = 'x',labelsize=17,rotation=65)

c.tick_params(axis = 'y',labelsize=17)



regressor_values=regressor_df['RMSLE deviation'].values

for i,index in enumerate(regressor_values):

    c.text(i,index, round(index,4), color='black', ha="center",fontsize=17)



regressor_df = regressor_df.sort_values(by='RMSLE', ascending = False)

    

b = sns.barplot(x = regressor_df['Regressor'],

                y=regressor_df['RMSLE'],palette="ch:.25", ax=ax1)

b.axes.set_title(" Score of the Models ",fontsize=20)

b.set_xlabel("Models",fontsize=17)

b.set_ylabel("Root Mean Square Error",fontsize=17)

b.tick_params(axis = 'x',labelsize=17,rotation=65)

b.tick_params(axis = 'y',labelsize=17)



regressor_values=regressor_df['RMSLE'].values

for i,index in enumerate(regressor_values):

    b.text(i,index, round(index,4), color='black', ha="center",fontsize=17)
# We copy this models for the voting algorithm because we need the unfitted models



lasso_r = lasso_reg

ridge_r = ridge_reg

elnet_r = elnet_reg

svr_r   = svr_reg

lgbm_r   = lgbm_reg



lasso_reg.fit(X,y) # Lasso 

elnet_reg.fit(X,y)  # Elastic Net

ridge_reg.fit(X,y) #Ridge

svr_reg.fit(X,y)   #SVR

lgbm_reg.fit(X,y) #LightGBM




def blended_predictions(x_data):

    return ((0.3 * lasso_reg.predict(x_data)) + (0.2 * elnet_reg.predict(x_data)) + (0.2 * ridge_reg.predict(x_data)) + \

            (0.2 * svr_reg.predict(x_data)) + (0.1 * lgbm_reg.predict(x_data)))

y_pred_blended = np.expm1(blended_predictions(test))

from sklearn.ensemble import VotingRegressor

# We need the unfitted models...

vot_reg = VotingRegressor([('stk', stk_reg),('lss', lasso_r),

                      ('rdg', ridge_r),('lnt', elnet_r),

                      ('svr', svr_r), ('lgbm', lgbm_r)])

score_voting= round(rmse_cv(vot_reg).mean(),5)

score_voting_std= round(rmse_cv(vot_reg).std(),5)

print("Voting regressor: "+ str(score_voting)+ ', std : '+str(score_voting_std))

vot_reg.fit(X,y)

sub_votting = np.expm1(vot_reg.predict(test))

sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = sub_votting

sub.to_csv('submission_Votting.csv',index=False)