import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



import seaborn as sns

from scipy import stats

from scipy.stats import norm, skew

from scipy.special import boxcox1p



pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)



from sklearn.impute import KNNImputer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import RobustScaler,StandardScaler

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold, cross_val_score, train_test_split,GridSearchCV

from sklearn.pipeline import make_pipeline



from xgboost import XGBRegressor

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from lightgbm import LGBMRegressor

from sklearn.svm import SVR

from mlxtend.regressor import StackingCVRegressor



import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn
df_train = pd.read_csv('../input/home-data-for-ml-course/train.csv')

df_test = pd.read_csv('../input/home-data-for-ml-course/test.csv')



test_id = df_test['Id']



n_train = df_train.shape[0]

n_test = df_train.shape[0]
df = pd.concat([df_train,df_test],sort = False).reset_index(drop = True)

df.shape
df.head()
df.info()
df.isnull().sum()
df.describe()
df['MSZoning'].fillna('RL',inplace = True)

df['Utilities'].fillna('AllPub',inplace = True)



#Exter

df['Exterior1st'].fillna('Plywood',inplace = True)

df['Exterior2nd'].fillna('Plywood',inplace = True)



#MasVnr

df['MasVnrType'][2610] = 'BrkFace'

df['MasVnrType'].fillna('None',inplace = True)



#Bsmt

df['BsmtQual'][2217] = 'Fa'

df['BsmtQual'][2218] = 'TA'

df['BsmtCond'][2040] = 'TA'

df['BsmtCond'][2185] = 'TA'

df['BsmtCond'][2524] = 'TA'

df['BsmtExposure'][948] = 'No'

df['BsmtExposure'][1487] = 'No'

df['BsmtExposure'][2348] = 'No'

df['BsmtFinType2'][332] = 'Rec'



df['BsmtQual'].fillna('NA',inplace = True)

df['BsmtCond'].fillna('NA',inplace = True)

df['BsmtExposure'].fillna('NA',inplace = True)

df['BsmtFinType1'].fillna('NA',inplace = True)

df['BsmtFinType2'].fillna('NA',inplace = True)



#electrical

df['Electrical'].fillna('SBrkr',inplace = True)



#FireplaceQu

df['FireplaceQu'].fillna('NA',inplace = True)



#kitchen

df['KitchenQual'].fillna('TA',inplace = True)



#Functional

df['Functional'].fillna('Typ',inplace = True)



##Garage

df['GarageType'][2576] = 'NA'

df['GarageYrBlt'][2126] = 1958

df['GarageFinish'][2126] = 'Unf'

df['GarageQual'][2126] = 'TA'

df['GarageCond'][2126] = 'TA'



df['GarageType'].fillna('NA',inplace = True)

df['GarageFinish'].fillna('NA',inplace = True)

df['GarageQual'].fillna('NA',inplace = True)

df['GarageCond'].fillna('NA',inplace = True)



df['SaleType'].fillna('WD',inplace = True)
df['MasVnrArea'].fillna(0,inplace = True)



#bsmt

df['BsmtFinSF1'].fillna(0,inplace = True)

df['BsmtFinSF2'].fillna(0,inplace = True)

df['BsmtUnfSF'].fillna(0,inplace = True)

df['TotalBsmtSF'].fillna(0,inplace = True)



#bsmt bath

df['BsmtFullBath'].fillna(0,inplace = True)

df['BsmtHalfBath'].fillna(0,inplace = True)





#garage

df['GarageYrBlt'].fillna(0,inplace = True)

df['GarageCars'].fillna(0,inplace = True)

df['GarageArea'].fillna(0,inplace = True)
def knn_imputer_nan(x,k):

    imputer = KNNImputer(n_neighbors=k)

    return pd.DataFrame(imputer.fit_transform(x))



df_LotFrontage_LotArea = df[['LotFrontage','LotArea']]

i_LotFrontage = knn_imputer_nan(df_LotFrontage_LotArea,36)

i_LotFrontage.iloc[:,0] = round(i_LotFrontage.iloc[:,0])

df['LotFrontage'] = i_LotFrontage.iloc[:,0]
Street_map = {'Grvl':0,'Pave':1}

Utilities_map = {'ELO':1,'NoSeWa':2,'NoSewr':3,'AllPub':4}

LandSlope_map = {'Sev':1,'Mod':2,'Gtl':3}

ExterQual_map = {'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

ExterCond_map = {'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5} 

BsmtQual_map = {'NA':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

BsmtCond_map = {'NA':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

BsmtExposure_map = {'NA':0,'No':1,'Mn':2,'Av':3,'Gd':4}

BsmtFinType1_map = {'NA':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6}

BsmtFinType2_map = {'NA':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6}

HeatingQC_map = {'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

KitchenQual_map = {'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

Functional_map = {'Sal':1,'Sev':2,'Maj2':3,'Maj1':4,'Mod':5,'Min2':6,'Min1':7,'Typ':8}

FireplaceQu_map = {'NA':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

GarageFinish_map = {'NA':0,'Unf':1,'RFn':2,'Fin':3}

GarageQual_map = {'NA':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

GarageCond_map = {'NA':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

PavedDrive_map = {'N':1,'P':2,'Y':3}



df.replace({'Street':Street_map,'Utilities':Utilities_map,'LandSlope':LandSlope_map,'ExterQual':ExterQual_map,

           'ExterCond':ExterCond_map,'BsmtQual':BsmtQual_map,'BsmtCond':BsmtCond_map,'BsmtExposure':BsmtExposure_map,

           'BsmtFinType1':BsmtFinType1_map,'BsmtFinType2':BsmtFinType2_map,'HeatingQC':HeatingQC_map,

           'KitchenQual':KitchenQual_map,'Functional':Functional_map,'FireplaceQu':FireplaceQu_map,'GarageFinish':GarageFinish_map,

           'GarageQual':GarageQual_map,'GarageCond':GarageCond_map,'PavedDrive':PavedDrive_map},inplace = True)



df['BsmtCond'] = df['BsmtCond'].apply(int)
#YearAdd

yr_feat = ['YearBuilt','YearRemodAdd']

for feature in yr_feat:

    df[feature] = df['YrSold'] - df[feature]

    

df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

df['Total_porch_SF'] = df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF']

df['Total_Bathrooms'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])

df['Total_Finished_SF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['BsmtFinSF1'] + df['BsmtFinSF2']



df['haspool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

df['has2ndfloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

df['hasgarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

df['hasbsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

df['hasfireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

df['hasMisc'] = df['MiscVal'].apply(lambda x: 1 if x > 0 else 0)
df['MSSubClass'] = df['MSSubClass'].apply(str)

df['YrSold'] = df['YrSold'].apply(str)

df['MoSold'] = df['MoSold'].apply(str)

cat_features = ['MSSubClass','MSZoning','LotShape','LandContour','LotConfig','Neighborhood','Condition1','Condition2',

                'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',

                'Foundation','Heating','CentralAir','Electrical','GarageType','MoSold','YrSold','SaleType','SaleCondition']



for feature in cat_features:

    df_ = pd.get_dummies(df[feature],prefix = feature,drop_first=True)

    df.drop(feature,inplace = True,axis = 1)

    df = pd.concat([df,df_],axis = 1)
df.drop(['Id','Alley','Utilities','Street','PoolArea','PoolQC','Fence','MiscFeature','MiscVal'],axis = 1,inplace=True)
df.head()
corrmat = df.iloc[:n_train,:].corr()

top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.6]

plt.figure(figsize=(10,10))

g = sns.heatmap(df.iloc[:n_train,:][top_corr_features].corr(),annot=True,cmap="RdYlGn")
# GrLivArea

plt.scatter(df['GrLivArea'][:n_train],df['SalePrice'][:n_train])

df = df.drop(df[((df['GrLivArea']>4000) & (df['SalePrice']<300000))].index)

n_train -= 2

plt.scatter(df['GrLivArea'][:n_train],df['SalePrice'][:n_train])



#OverallQual

# sns.boxplot(df['OverallQual'],df['SalePrice'])



#ExterQual

# sns.boxplot(df['ExterQual'],df['SalePrice'])
#SalePrice 

sns.distplot(df['SalePrice'][:n_train],fit = norm)



pl = plt.figure()

stats.probplot(df['SalePrice'][:n_train],plot=plt)

plt.show()
df['SalePrice'][:n_train] = np.log1p(df['SalePrice'][:n_train])



sns.distplot(df['SalePrice'][:n_train] , fit=norm);



pl = plt.figure()

stats.probplot(df['SalePrice'][:n_train],plot=plt)

plt.show()
continuous_features = [feature for feature in df.columns if len(df[feature].unique())>10 and df[feature].dtype != 'object' and 'Year' not in feature and 'Yr' not in feature]

continuous_features,len(continuous_features)
continuous_features.remove('SalePrice')

sk = df[continuous_features].apply(lambda x:skew(x)).sort_values(ascending = False)

sk = pd.DataFrame(sk)

sk
ch = [0,0.03,0.05,0.08,0.1,0.13,0.15]

df__ = pd.DataFrame()

for choice in ch:

    df_ = pd.DataFrame(skew(boxcox1p(df[continuous_features],choice)),columns=[choice],index = continuous_features)

    df__ = pd.concat([df__,df_],axis = 1)

    

df__ = pd.concat([pd.DataFrame(skew(df[continuous_features]),columns = ['Org'],index = continuous_features),df__],axis = 1)





skew_result = {}

for i in df__.index:

    min_ = 'Org'

    for j in df__.columns:

        if df__.loc[i,j]>=0 and df__.loc[i,j]<df__.loc[i,min_]:

            min_ = j

            

    skew_result[i] = min_

    



print(skew_result)

skew_result = {k:v for k,v in skew_result.items() if v != 'Org'}
#boxcox1p for other continuous values 

for k,v in skew_result.items():

    df[k] = boxcox1p(df[k],v)
df_train = df.iloc[:n_train,:]

df_test = df.iloc[n_train:,:]



x = df_train.drop('SalePrice',axis = 1)

y = df_train['SalePrice']

df_test.drop('SalePrice',inplace = True,axis = 1)
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
#Validation

kf = KFold(5, shuffle=True, random_state=42)



def rmse(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def rmsle_cv(model):

    rmse= np.sqrt(-cross_val_score(model, x.values, y.values, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
lgbm = LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.01, n_estimators=5000,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.01, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=5000,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             nthread = -1)



gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber')



rf = RandomForestRegressor(n_estimators=1200,

                          max_depth=15,

                          min_samples_split=5,

                          min_samples_leaf=5,

                          max_features=None,

                          oob_score=True)

                          





alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]



ridge = make_pipeline(RobustScaler(),

                      RidgeCV(alphas=alphas_alt, cv=kf))



lasso = make_pipeline(RobustScaler(),

                      LassoCV(max_iter=1e7, alphas=alphas2,

                              random_state=42, cv=kf))



elasticnet = make_pipeline(RobustScaler(),

                           ElasticNetCV(max_iter=1e7, alphas=e_alphas,

                                        cv=kf, l1_ratio=e_l1ratio))





stack_gen = StackingCVRegressor(regressors=(xgb, lgbm, gbr, rf,ridge,lasso,elasticnet),

                                meta_regressor=xgb,

                                use_features_in_secondary=True,

                                cv = 5)

                              

                                