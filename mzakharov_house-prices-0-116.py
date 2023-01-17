# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from datetime import datetime

from scipy.stats import skew  # for some statistics

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from mlxtend.regressor import StackingCVRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt

import scipy.stats as stats

import sklearn.linear_model as linear_model

import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

import os, sys, gc, warnings, random, datetime, math

from sklearn.preprocessing import LabelEncoder



#import os

#print(os.listdir("../input"))



import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
def make_test_predictions(tr_df, tt_df, target, lgb_params):#, NFOLDS=2):

    

    #new_columns = set(list(train_df)).difference(base_columns + remove_features)

    #features_columns = base_columns + list(new_columns)

    features_columns = tr_df.drop([target], axis=1).columns

    

    #folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)



    X,y = tr_df[features_columns], tr_df[target]    

    P,P_y = tt_df[features_columns], tt_df[target]



    for col in list(X):

        if X[col].dtype=='O':

            X[col] = X[col].fillna('unseen_before_label')

            P[col] = P[col].fillna('unseen_before_label')



            X[col] = train_df[col].astype(str)

            P[col] = test_df[col].astype(str)



            le = LabelEncoder()

            le.fit(list(X[col])+list(P[col]))

            X[col] = le.transform(X[col])

            P[col]  = le.transform(P[col])



            X[col] = X[col].astype('category')

            P[col] = P[col].astype('category')

        

    tt_df = tt_df[['Id',target]]    

    predictions = np.zeros(len(tt_df))



    tr_data = lgb.Dataset(X, label=y)

    vl_data = lgb.Dataset(P, label=P_y) 

    estimator = lgb.train(

            lgb_params,

            tr_data,

            valid_sets = [tr_data, vl_data],

            verbose_eval = 200,

        )   

        

    pp_p = estimator.predict(P)

    #predictions += pp_p/NFOLDS



    feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(),X.columns)), columns=['Value','Feature'])

    print(feature_imp)

        

    #tt_df['prediction'] = predictions

    tt_df['prediction'] = pp_p

    

    return tt_df

## -------------------
def frequency_encoding(train_df, test_df, columns, self_encoding=False):

    for col in columns:

        temp_df = pd.concat([train_df[[col]], test_df[[col]]])

        fq_encode = temp_df[col].value_counts(dropna=False).to_dict()

        if self_encoding:

            train_df[col] = train_df[col].map(fq_encode)

            test_df[col]  = test_df[col].map(fq_encode)            

        else:

            train_df[col+'_fq_enc'] = train_df[col].map(fq_encode)

            test_df[col+'_fq_enc']  = test_df[col].map(fq_encode)

    return train_df, test_df
########################### Model params

lgb_params = {

                    'objective':'regression',

                    'boosting_type':'gbdt',

                    'metric':'rmse',

                    'n_jobs':-1,

                    'learning_rate':0.01,

                    'num_leaves': 2**8,

                    'max_depth':-1,

                    'tree_learner':'serial',

                    'colsample_bytree': 0.7,

                    'subsample_freq':1,

                    'subsample':0.7,

                    'n_estimators':80000,

                    'max_bin':255,

                    'verbose':-1,

                    'seed': 42,

                    'early_stopping_rounds':100, 

                } 
TARGET = 'SalePrice'
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

print ("Data is loaded!")
print ("Train: ",train.shape[0],"sales, and ",train.shape[1],"features")

print ("Test: ",test.shape[0],"sales, and ",test.shape[1],"features")
train.head()
test.head()
quantitative = [f for f in train.columns if train.dtypes[f] != 'object']

quantitative.remove('SalePrice')

quantitative.remove('Id')

qualitative = [f for f in train.columns if train.dtypes[f] == 'object']
print(quantitative)
print(qualitative)
sns.set_style("whitegrid")

missing = train.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()
y = train['SalePrice']

plt.figure(1); plt.title('Johnson SU')

sns.distplot(y, kde=False, fit=stats.johnsonsu)

plt.figure(2); plt.title('Normal')

sns.distplot(y, kde=False, fit=stats.norm)

plt.figure(3); plt.title('Log Normal')

sns.distplot(y, kde=False, fit=stats.lognorm)
test_normality = lambda x: stats.shapiro(x.fillna(0))[1] < 0.01

normal = pd.DataFrame(train[quantitative])

normal = normal.apply(test_normality)

print(not normal.any())
def encode(frame, feature):

    ordering = pd.DataFrame()

    ordering['val'] = frame[feature].unique()

    ordering.index = ordering.val

    ordering['spmean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice'] 

    ordering = ordering.sort_values('spmean')

    ordering['ordering'] = range(1, ordering.shape[0]+1)

    ordering = ordering['ordering'].to_dict()

    

    for cat, o in ordering.items():

        frame.loc[frame[feature] == cat, feature+'_E'] = o

    

qual_encoded = [] # for qualitative features heatmap 

for q in qualitative:  

    encode(train, q)

    qual_encoded.append(q+'_E')

print(qual_encoded)
plt.figure(1,figsize=(12, 9))

corr = train[quantitative+['SalePrice']].corr()

sns.heatmap(corr)

plt.figure(2,figsize=(12, 9))

corr = train[qual_encoded+['SalePrice']].corr()

sns.heatmap(corr)
train['OverallQual'].unique()
train['Neighborhood'].unique()
fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'], color = 'lightgreen', edgecolor='black')

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

#train.drop(['Id'], axis=1, inplace=True)

train = train[train.GrLivArea < 4500] # exclude outliers

train.reset_index(drop=True, inplace=True)

train["SalePrice"] = np.log1p(train["SalePrice"])



features = train

train_df = features[:1000]

test_df = features[1000:]



del train

gc.collect()



features.shape
plt.figure(2); plt.title('Normal')

sns.distplot(features["SalePrice"], kde=False, fit=stats.norm)
import lightgbm as lgb

test_predictions = make_test_predictions(train_df, test_df, TARGET, lgb_params)

print("RMSE:", np.sqrt(mean_squared_error(test_predictions[TARGET], test_predictions['prediction'])))
#Change feature type and fillna for categorical features

features['MSSubClass'] = features['MSSubClass'].apply(str) # to categorical

features['YrSold'] = features['YrSold'].astype(str) # to categorical

features['MoSold'] = features['MoSold'].astype(str) # to categorical

features['Functional'] = features['Functional'].fillna('Typ') 

features['Electrical'] = features['Electrical'].fillna("SBrkr") 

features['KitchenQual'] = features['KitchenQual'].fillna("TA") 

features["PoolQC"] = features["PoolQC"].fillna("None")

features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0]) # fillna by most common value

features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])

features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

#More fillna's

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    features[col] = features[col].fillna(0)



for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    features[col] = features[col].fillna('None')



for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    features[col] = features[col].fillna('None')
# MSSubClass: The building class

# MSZoning: The general zoning classification

# fillna row's 'MSZoning' by most common value for its MSSubClass 

features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
# fillna row's 'LotFrontage'feature by median value for its 'Neighborhood' 

features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
#for all other categorical:

objects = []

for i in features.columns:

    if features[i].dtype == object:

        objects.append(i)

features.update(features[objects].fillna('None'))



#for all other numerical:

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics = []

for i in features.columns:

    if features[i].dtype in numeric_dtypes:

        numerics.append(i)

features.update(features[numerics].fillna(0))
#transform quantitative to normal distribution

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics2 = []

for i in features.columns:

    if features[i].dtype in numeric_dtypes:

        numerics2.append(i)

skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)



high_skew = skew_features[skew_features > 0.5]

skew_index = high_skew.index



for i in skew_index:

    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))
# separate features for test-validation

train_df = features[:1000]

test_df = features[1000:]

test_predictions = make_test_predictions(train_df, test_df, TARGET, lgb_params)

print("RMSE:", np.sqrt(mean_squared_error(test_predictions[TARGET], test_predictions['prediction'])))
#drop features with nunique()<5

features = features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)



#Combine

features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']

#total area:

features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']

#total living area:

features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +

                                 features['1stFlrSF'] + features['2ndFlrSF'])



features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +

                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))



features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +

                              features['EnclosedPorch'] + features['ScreenPorch'] +

                              features['WoodDeckSF'])

#create some binarized features

features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
train_df = features[:1000]

test_df = features[1000:]

test_predictions = make_test_predictions(train_df, test_df, TARGET, lgb_params)

print("RMSE:", np.sqrt(mean_squared_error(test_predictions[TARGET], test_predictions['prediction'])))
# In my mind it's easier to predict sale price if we know average price per square feet in each location with

# correction on property's material and finish quality (OverallQual).

# For this approach we will create 2 new features:

# 1) 'Price per square feet' for each lot by using its 'SalePrice' and 'LotArea'

# 2) 'Ngbhd_oql_av_psqf' - calculate average price per square feet for 'Neighborhood' groups separated by 'OverallQual' for 

# using on test set
features['Price_per_sqf'] = features['SalePrice'] / features['LotArea']

features['Ngbhd_oql_av_psqf'] = features.groupby(['Neighborhood','OverallQual'])['Price_per_sqf'].transform(lambda x: x.mean())

# ,'OverallQual'

features.drop(['Price_per_sqf'], axis=1, inplace = True) #cause overfitting by himself
train_df = features[:1000]

test_df = features[1000:]

test_predictions = make_test_predictions(train_df, test_df, TARGET, lgb_params)

print("RMSE:", np.sqrt(mean_squared_error(test_predictions[TARGET], test_predictions['prediction'])))
#frequency_encoding:

#columns = ['TotalSF', 'LotArea', 'Total_sqr_footage', 'GrLivArea', 'GarageArea', 'Total_porch_sf']

columns = ['OverallQual', 'Neighborhood'] #for most important features

train_df, test_df = frequency_encoding(train_df, test_df, columns, self_encoding=False)
test_predictions = make_test_predictions(train_df, test_df, TARGET, lgb_params)

print("RMSE:", np.sqrt(mean_squared_error(test_predictions[TARGET], test_predictions['prediction'])))
del train_df, test_df, features

gc.collect()
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

print ("Data is loaded!")
train.drop(['Id'], axis=1, inplace=True)# Id columns can misslead our model

test.drop(['Id'], axis=1, inplace=True)

train.shape, test.shape
train = train[train.GrLivArea < 4500]

train.reset_index(drop=True, inplace=True)

y = np.log1p(train["SalePrice"])

#y = train['SalePrice'].reset_index(drop=True)

train.shape
train_features = train.drop(['SalePrice'], axis=1)

test_features = test



features = pd.concat([train_features, test_features]).reset_index(drop=True)

del train,test, train_features, test_features

gc.collect()

features.shape
#Change feature type and fillna for categorical features

features['MSSubClass'] = features['MSSubClass'].apply(str) # to categorical

features['YrSold'] = features['YrSold'].astype(str) # to categorical

features['MoSold'] = features['MoSold'].astype(str) # to categorical

features['Functional'] = features['Functional'].fillna('Typ') 

features['Electrical'] = features['Electrical'].fillna("SBrkr") 

features['KitchenQual'] = features['KitchenQual'].fillna("TA") 

features["PoolQC"] = features["PoolQC"].fillna("None")

features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0]) # fillna by most common value

features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])

features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])



##################More fillna's

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    features[col] = features[col].fillna(0)



for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    features[col] = features[col].fillna('None')



for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    features[col] = features[col].fillna('None')



############### MSSubClass: The building class

# MSZoning: The general zoning classification

# fillna row's 'MSZoning' by most common value for its MSSubClass 

features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))



############### fillna row's 'LotFrontage'feature by median value for its 'Neighborhood' 

features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

       

##################for all other categorical:

objects = []

for i in features.columns:

    if features[i].dtype == object:

        objects.append(i)

features.update(features[objects].fillna('None'))



###################for all other numerical:

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics = []

for i in features.columns:

    if features[i].dtype in numeric_dtypes:

        numerics.append(i)

features.update(features[numerics].fillna(0))



#######################transform quantitative to normal distribution

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics2 = []

for i in features.columns:

    if features[i].dtype in numeric_dtypes:

        numerics2.append(i)

skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)



high_skew = skew_features[skew_features > 0.75]

skew_index = high_skew.index



for i in skew_index:

    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))



################drop features with nunique()<5

features = features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)



#Combine

features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']

#total area:

features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']

#total living area:

features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +

                                 features['1stFlrSF'] + features['2ndFlrSF'])



features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +

                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))



features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +

                              features['EnclosedPorch'] + features['ScreenPorch'] +

                              features['WoodDeckSF'])



######################create some binarized features

features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)



features.shape
len(y)
# Separate features-df backwards beacause we can't calculate 'Price_per_sqf' for test set

train_df = features[:len(y)]

test_df = features[len(y):]
# crate 2 features for train like on local test beyond

train_df['Price_per_sqf'] = y / train_df['LotArea']

train_df['Ngbhd_oql_av_psqf'] = train_df.groupby(['Neighborhood','OverallQual'])['Price_per_sqf'].transform(lambda x: x.mean())

train_df.drop(['Price_per_sqf'], axis=1, inplace = True)
# empty column for filling

test_df['Ngbhd_oql_av_psqf']=np.nan
#ugly script for pulling our feature from train set to test



test_df_ind_drop = test_df.reset_index(drop=True) # temporary test-df



from tqdm import tqdm_notebook



for i,rowa in enumerate(tqdm_notebook(list(test_df_ind_drop.values))):

    for j,rowb in enumerate(train_df.values):

        if rowb[train_df.columns.get_loc('Neighborhood')]==rowa[test_df_ind_drop.columns.get_loc('Neighborhood')] and rowb[train_df.columns.get_loc('OverallQual')]==rowa[test_df_ind_drop.columns.get_loc('OverallQual')]:

            test_df_ind_drop['Ngbhd_oql_av_psqf'][i] = train_df['Ngbhd_oql_av_psqf'][j]

test_df_ind_drop[:100]
#restore indecies

old_indecies = test_df.index 

test_df = test_df_ind_drop.set_index(old_indecies)
# estimate number of missing values



def missing(df, column):

    missing = df[column].isnull().sum()

    return (missing)



missing(test_df, 'Ngbhd_oql_av_psqf')
save_train_df = train_df

save_test_df = test_df
#train_df = save_train_df

#test_df = save_test_df
#fillna remaining Nan's in 'Ngbhd_oql_av_psqf'("Neighborhood overall quality average price per square feet") 

#with mean values for each Neighborhood group

test_df['Ngbhd_oql_av_psqf'] = test_df.groupby('Neighborhood')['Ngbhd_oql_av_psqf'].transform(lambda x: x.fillna(x.mean()))

test_df[:100]
# check missing

missing(test_df, 'Ngbhd_oql_av_psqf')
features = pd.concat([train_df, test_df])
final_features = pd.get_dummies(features).reset_index(drop=True)

final_features.shape
X = final_features.iloc[:len(y), :]

X_sub = final_features.iloc[len(y):, :]

X.shape, y.shape, X_sub.shape
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)



# rmsle

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



# build our model scoring function

def cv_rmse(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)
# setup models    

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]



ridge = make_pipeline(RobustScaler(),

                      RidgeCV(alphas=alphas_alt,

                      cv=kfolds))



lasso = make_pipeline(RobustScaler(),

                      LassoCV(max_iter=1e7, 

                              alphas=alphas2,

                              random_state=42, 

                              cv=kfolds

                             )

                     )



elasticnet = make_pipeline(RobustScaler(),

                           ElasticNetCV(max_iter=1e7, 

                                        alphas=e_alphas,

                                        cv=kfolds, 

                                        l1_ratio=e_l1ratio

                                       )

                          )

                                        

svr = make_pipeline(RobustScaler(),

                    SVR(C= 20, epsilon= 0.008, gamma=0.0003,))





gbr = GradientBoostingRegressor(n_estimators=3000, 

                                learning_rate=0.05,

                                max_depth=4, 

                                max_features='sqrt',

                                min_samples_leaf=15, 

                                min_samples_split=10, 

                                loss='huber', 

                                random_state =42

                               )

                                   



lightgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=4,

                                       learning_rate=0.01, 

                                       n_estimators=5000,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose=-1,

                                       #min_data_in_leaf=2,

                                       #min_sum_hessian_in_leaf=11

                                       )

                                       



xgboost = XGBRegressor(learning_rate=0.01, 

                       n_estimators=3460,

                       max_depth=3, 

                       min_child_weight=0,

                       gamma=0, 

                       subsample=0.7,

                       colsample_bytree=0.7,

                       objective='reg:linear', 

                       nthread=-1,

                       scale_pos_weight=1, 

                       seed=27,

                       reg_alpha=0.00006

                      )



# stacking model

stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet,

                                            gbr, xgboost, lightgbm),

                                meta_regressor=lightgbm,

                                use_features_in_secondary=True)
print('TEST score on CV')



score = cv_rmse(ridge)

print("Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.datetime.now(), )



score = cv_rmse(lasso)

print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.datetime.now(), )



score = cv_rmse(elasticnet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.datetime.now(), )



score = cv_rmse(svr)

print("SVR score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.datetime.now(), )



score = cv_rmse(lightgbm)

print("Lightgbm score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.datetime.now(), )



score = cv_rmse(gbr)

print("GradientBoosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.datetime.now(), )



score = cv_rmse(xgboost)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.datetime.now(), )
print('START Fit')

print(datetime.datetime.now(), 'StackingCVRegressor')

stack_gen_model = stack_gen.fit(np.array(X), np.array(y))

print(datetime.datetime.now(), 'elasticnet')

elastic_model_full_data = elasticnet.fit(X, y)

print(datetime.datetime.now(), 'lasso')

lasso_model_full_data = lasso.fit(X, y)

print(datetime.datetime.now(), 'ridge')

ridge_model_full_data = ridge.fit(X, y)

print(datetime.datetime.now(), 'svr')

svr_model_full_data = svr.fit(X, y)

print(datetime.datetime.now(), 'GradientBoosting')

gbr_model_full_data = gbr.fit(X, y)

print(datetime.datetime.now(), 'xgboost')

xgb_model_full_data = xgboost.fit(X, y)

print(datetime.datetime.now(), 'lightgbm')

lgb_model_full_data = lightgbm.fit(X, y)
def blend_models_predict(X):

    return ((0.15 * elastic_model_full_data.predict(X)) + \

            (0.15 * lasso_model_full_data.predict(X)) + \

            (0.15 * ridge_model_full_data.predict(X)) + \

            (0.15 * svr_model_full_data.predict(X)) + \

            (0.1 * gbr_model_full_data.predict(X)) + \

            (0.1 * xgb_model_full_data.predict(X)) + \

            (0.1 * lgb_model_full_data.predict(X)) + \

            (0.1 * stack_gen_model.predict(np.array(X))))

            

print('RMSLE score on train data:')

print(rmsle(y, blend_models_predict(X)))
print('Predict submission', datetime.datetime.now(),)

submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission.iloc[:,1] = np.floor(np.expm1(blend_models_predict(X_sub)))
submission.to_csv("new_submission123.csv", index=False)

print('Save submission', datetime.datetime.now(),)
submission.head()