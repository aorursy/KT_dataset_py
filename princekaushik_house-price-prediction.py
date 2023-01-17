import numpy as np 

import pandas as pd



import warnings

warnings.filterwarnings('ignore')



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

import matplotlib.pyplot as plt

import seaborn as sns 

import missingno as msno

%matplotlib inline

pd.set_option('display.max_rows',200)

pd.set_option('display.max_columns',100)

import pickle

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import RidgeCV

from sklearn.linear_model import LassoCV

from sklearn.linear_model import ElasticNetCV

from sklearn.svm import SVR

from sklearn.ensemble import GradientBoostingRegressor

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

from mlxtend.regressor import StackingCVRegressor

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error,make_scorer

from sklearn.model_selection import KFold,RandomizedSearchCV

import statsmodels.formula.api as smf

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

print(train.shape)

print(test.shape)
train.info()
# Combining dataset

train_y = train['SalePrice']

data = pd.concat((train,test),sort= False).reset_index(drop=True)

data.drop(['SalePrice','Id'],axis=1,inplace=True)

data.rename(columns={'1stFlrSF':'FirstFlrSF','2ndFlrSF':'SecondFlrSF','3SsnPorch':'ThreeSsnPorch'}, inplace=True)

data.shape
data.head()
train_y.describe()
#Distribution plot

sns.distplot(train_y);

#skewness and Kurtosis

print('Skewness: %f' % train_y.skew())

print('Kurtosis: %f' % train_y.kurt())
# using numpy function log fucntion

train_y = np.log(train_y + 1)

sns.distplot(train_y);

print("Skewness: %f" % train_y.skew())

print("Kurtosis: %f" % train_y.kurt())
# correlation matrix

plt.subplots(figsize=(10,8))

sns.heatmap(train.corr());
#Saleplice corr matrix

k = 10# no. of variables for heatmap

cols = train.corr().nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale = 1)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# Top

num = train.corr()['SalePrice'].sort_values(ascending = False).head(10).to_frame()

cm = sns.light_palette('grey', as_cmap = True)

s = num.style.background_gradient(cmap = cm)

s
# Compute the correlation matrix 

corr_all = train.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr_all, dtype = np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize = (11, 9))



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr_all, mask = mask,

            square = True, linewidths = .5, ax = ax, cmap = "BuPu")      

plt.show()
#scatter plot

sns.set()

sns.pairplot(train[cols],size = 2.5);
# missing values?

sns.set(style = "ticks")



msno.matrix(data)

msno.heatmap(data, cmap = 'binary')
missing_data = pd.DataFrame(data.isnull().sum()).reset_index()

missing_data.columns = ['ColumnName','MissingCount']



missing_data['PercentMissing'] = round(missing_data['MissingCount']/data.shape[0],3)*100

missing_data =missing_data.sort_values(by = 'MissingCount',ascending = False).reset_index(drop = True)

missing_data.head(35)
data.drop(['PoolQC','MiscFeature','Alley'],axis=1,inplace=True)

ffill= list(missing_data.ColumnName[18:34])

data[ffill] = data[ffill].fillna(method = 'ffill')

missing_data.ColumnName[3:18]
col_for_zero = ['Fence','FireplaceQu','GarageFinish','GarageQual','GarageCond','GarageType','BsmtExposure','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1','MasVnrType']

data[col_for_zero] = data[col_for_zero].fillna('None')

data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].dropna().mean())

data['GarageYrBlt'] = data['GarageYrBlt'].fillna(data['GarageYrBlt'].dropna().median())

data['MasVnrArea'] = data['MasVnrArea'].fillna(data['MasVnrArea'].dropna().median())
data['YrBltAndRemod']=data['YearBuilt']+data['YearRemodAdd']

data['TotalSF']=data['TotalBsmtSF'] + data['FirstFlrSF'] + data['SecondFlrSF']

data['Total_sqr_footage'] = (data['BsmtFinSF1'] + data['BsmtFinSF2'] +data['FirstFlrSF'] + data['SecondFlrSF'])

data['Total_Bathrooms'] = (data['FullBath'] + (0.5 * data['HalfBath']) +data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath']))

data['Total_porch_sf'] = (data['OpenPorchSF'] + data['ThreeSsnPorch'] + data['EnclosedPorch'] + data['ScreenPorch'] + data['WoodDeckSF'])

data['hasfence'] = data['Fence'].apply(lambda x: 0 if x == 0 else 1).astype(str)

data['hasmasvnr'] = data['MasVnrArea'].apply(lambda x: 0 if x == 0 else 1).astype(str)

data['haspool'] = data['PoolArea'].apply(lambda x: 1 if x > 0 else 0).astype(str)

data['has2ndfloor'] = data['SecondFlrSF'].apply(lambda x: 1 if x > 0 else 0).astype(str)

data['hasgarage'] = data['GarageArea'].apply(lambda x: 1 if x > 0 else 0).astype(str)

data['hasbsmt'] = data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0).astype(str)

data['hasfireplace'] = data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0).astype(str)

data['MSSubClass'] = data['MSSubClass'].astype(str)

data['YrSold'] = data['YrSold'].astype(str)

data['MoSold'] = data['MoSold'].astype(str)
num_var=[key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['float64', 'int64', 'float32', 'int32']]

cat_var=[key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['object']]

print(len(num_var))

print(len(cat_var))

num_data = data[num_var]

cat_data = data[cat_var]
#skew X variables

skew_data = num_data.apply(lambda x: x.skew()).sort_values(ascending=False)





high_skew = skew_data[skew_data > 0.5]

skew_index = high_skew.index



for i in skew_index:

    data[i] = boxcox1p(data[i], boxcox_normmax(data[i] + 1))
def outlier_capping(x):

    x = x.clip_upper(x.quantile(0.99))

    x = x.clip_lower(x.quantile(0.01))

    return x

num_data.drop('PoolArea',axis=1,inplace=True)

num_data = num_data.apply(outlier_capping)

num_data['PoolArea'] = data.PoolArea
def create_dummies(df,colname):

    col_dummies = pd.get_dummies(df[colname],prefix =colname)

    col_dummies.drop(col_dummies.columns[0],axis=1,inplace=True)

    df = pd.concat([df,col_dummies],axis=1)

    df.drop(colname,axis=1,inplace=True)

    return df

#for c_feature in categorical_features

for c_feature in cat_data.columns:

    cat_data[c_feature] = cat_data[c_feature].astype('category')

    cat_data = create_dummies(cat_data , c_feature )

print(cat_data.shape)

print(num_data.shape)
final_data = pd.concat([cat_data,num_data,train_y],axis=1)

print(final_data.shape)
final_data.columns= [var.strip().replace('.', '_') for var in final_data.columns]

final_data.columns= [var.strip().replace('&', '_') for var in final_data.columns]

final_data.columns= [var.strip().replace(' ', '_') for var in final_data.columns]
overfit = []

for i in final_data.columns:

    counts = final_data[i].value_counts()

    zeros = counts.iloc[0]

    if zeros / len(final_data) * 100 > 99.94:

        overfit.append(i)



overfit

final_data.drop(overfit,axis=1,inplace=True)
#splitting the data set into two sets

final_train = final_data.loc[final_data.SalePrice.isnull()==0]

final_test = final_data.loc[final_data.SalePrice.isnull()==1]

final_train = final_train.drop('SalePrice',axis=1)

final_test = final_test.drop('SalePrice',axis=1)

print(final_train.shape)

print(final_test.shape)
X = final_train

y = train_y

print(X.shape)

print(y.shape)

test_X = final_test

print(test_X.shape)
kfolds =  KFold(n_splits = 10,shuffle = True,random_state = 21)

def rmse(y,y_pred):

    return np.sqrt(mean_squared_error(y,y_pred))



def cv_rmse(model,X=X):

    rmse = np.sqrt(-cross_val_score(model,X,y,scoring='neg_mean_squared_error',cv = kfolds))

    return rmse
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5,15.6,15.7,15.8,15.9,16]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008,0.0009,0.0010,0.0011,0.0012,0.0013,0.0014,0.0015]

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008,0.0009,0.0010,0.0011,0.0012,0.0013,0.0014,0.0015]

e_l1ratio = [0.05, 0.15,0.2, 0.25,0.3, 0.35,0.4, 0.45,0.5, 0.55,0.6, 0.65,0.7, 0.75,0.8, 0.85, 0.9, 0.95, 0.99, 1]
# ridge = make_pipeline(RobustScaler(),RidgeCV(alphas = alphas_alt,cv = kfolds))

# lasso = make_pipeline(RobustScaler(),LassoCV(max_iter = 1e7,alphas = alphas2,random_state = 42,cv = kfolds))

# elasticnet = make_pipeline(RobustScaler(),ElasticNetCV(max_iter = 1e7,alphas = e_alphas,cv = kfolds,l1_ratio =e_l1ratio))

# svr = make_pipeline(RobustScaler(),SVR(C= 20,epsilon = 0.008 , gamma = 0.0003))

# gbr = GradientBoostingRegressor(n_estimators=3000,learning_rate= 0.05,max_depth=4,max_features='sqrt',min_samples_leaf = 15,min_samples_split = 10,loss = 'huber',random_state =21)

# lightgbm = LGBMRegressor(objective = 'regression',num_leaves = 4,learning_rate = 0.01,n_estimators=5000,max_bin = 200,bagging_fraction = 0.75,bagging_freq = 5,bagging_seed=7,feature_fraction=0.2,feature_fraction_seed = 7,verbose=-1)

# xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,max_depth=3,min_child_weight=0,gamma = 0,subsample=0.7,colsample_bytree=0.7,objective='reg:linear',nthread=-1,scale_pos_weight=1,seed=27,reg_alpha = 0.00006)

# stack_gen = StackingCVRegressor(regressors=(ridge,lasso,elasticnet,gbr,xgboost,lightgbm,svr),meta_regressor=xgboost,use_features_in_secondary=True)
# score = cv_rmse(ridge)

# print('Ridge: {:.4f} ({:.4f})\n'.format(score.mean(),score.std()))

# score = cv_rmse(lasso)

# print('Lasso: {:.4f} ({:.4f})\n'.format(score.mean(),score.std()))

# score = cv_rmse(elasticnet)

# print('Elasticnet: {:.4f} ({:.4f})\n'.format(score.mean(),score.std()))

# score = cv_rmse(svr)

# print('SVR: {:.4f} ({:.4f})\n'.format(score.mean(),score.std()))

# score = cv_rmse(gbr)

# print('GBRegressor: {:.4f} ({:.4f})\n'.format(score.mean(),score.std()))

# score = cv_rmse(lightgbm)

# print('LightGBM: {:.4f} ({:.4f})\n'.format(score.mean(),score.std()))

# score = cv_rmse(xgboost)

# print('XGBoost: {:.4f} ({:.4f})\n'.format(score.mean(),score.std()))
# print('Start Fit')



# print('stack_gen')

# stack_gen_model = stack_gen.fit(np.array(X),np.array(y))

# print('Ridge')

# ridge_model_full_data = ridge.fit(X,y)

# print('Lasso')

# lasso_model_full_data = lasso.fit(X,y)

# print('Elasticnet')

# elastic_model_full_data = elasticnet.fit(X,y)

# print('SVR')

# svr_model_full_data = svr.fit(X,y)

# print('GradientBoosting')

# gbr_model_full_data = gbr.fit(X,y)

# print('LightGBM')

# lightgbm_model_full_data = lightgbm.fit(X,y)

# print('XGBoost')

# xgboost_model_full_data = xgboost.fit(X,y)
# #Blending models

# def blend_models_predict(X):

#     return ((0.0175 * elastic_model_full_data.predict(X)) + \

#             (0.0175 * lasso_model_full_data.predict(X)) + \

#             (0.0075 * ridge_model_full_data.predict(X)) + \

#             (0.0075 * svr_model_full_data.predict(X)) + \

#             (0.2 * gbr_model_full_data.predict(X)) + \

#             (0.2 * xgboost_model_full_data.predict(X)) + \

#             (0.2 * lightgbm_model_full_data.predict(X)) + \

#             (0.35 * stack_gen_model.predict(np.array(X))))
# print('RMSE score on train data: ')    

# print(rmse(y,blend_models_predict(X)))
# print('Predict submission')

submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

# submission.iloc[:,1] = np.floor(np.expm1(blend_models_predict(test_X)))

submission = pd.read_csv('../input/topsubmissions/submission.csv')
# print('Blend with my Top Kernels submissions\n')

# sub_1 = pd.read_csv('../input/topsubmissions/submission1.csv')

# sub_2 = pd.read_csv('../input/topsubmissions/submission2.csv')



# submission.iloc[:,1] = np.floor((0.5 * np.floor(np.expm1(blend_models_predict(test_X)))) + 

                                

#                                 (0.5 * sub_2.iloc[:,1]))



submission.to_csv("submission.csv", index=False)

submission.head()