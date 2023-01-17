import numpy as np

import pandas as pd

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt

from sklearn import linear_model

from sklearn import preprocessing

from sklearn import metrics

from scipy import stats

from scipy.stats import norm, skew

from sklearn.model_selection import RandomizedSearchCV



sns.set_style('darkgrid')

plt.style.use("dark_background") 



import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)





pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

pd.set_option('display.width', None)

pd.set_option('display.max_colwidth', -1)
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train.shape
trainID = train.Id

testID = test.Id



train.drop('Id', axis = 1, inplace=True)

test.drop('Id', axis = 1, inplace=True)



train.shape
plt.figure(figsize=(14,10))



plt.subplot(2,2,1)

g = sns.scatterplot(x = 'GrLivArea', y = 'SalePrice', data = train)

plt.title('Before removing outliers')



train = train[((train.GrLivArea<4000) | (train.SalePrice>200000))]



plt.subplot(2,2,2)

plt.title('After removing outliers')

g = sns.scatterplot(x = 'GrLivArea', y = 'SalePrice', data = train)



plt.subplot(2,2,3)

g = sns.scatterplot(x = 'OverallQual', y = 'SalePrice', data = train)

plt.title('Before removing outliers')



train = train[((train.OverallQual>5) | (train.SalePrice<200000))]

train = train[((train.OverallQual>=9) | (train.SalePrice<500000))]



plt.subplot(2,2,4)

plt.title('After removing outliers')

g = sns.scatterplot(x = 'OverallQual', y = 'SalePrice', data = train)
g = sns.distplot(train.SalePrice, fit = norm)

(mu, sigma) = norm.fit(train.SalePrice)

print('mu = ', mu, ', sigma = ', sigma)



plt.figure()

g = stats.probplot(train.SalePrice, plot = plt)
train.SalePrice = np.log(train.SalePrice)

g = sns.distplot(train.SalePrice, fit = norm)



plt.figure()

g = stats.probplot(train.SalePrice, plot = plt)
ntrain = train.shape[0]

ntest = test.shape[0]

y = train.SalePrice.values

full_data = pd.concat((train,test)).reset_index(drop=True)

full_data.drop('SalePrice', axis = 1, inplace=True)

full_data.shape



#concatenate train and test rows
#which features have maximum missing values?

full_data.isnull().sum().sort_values(ascending=False).head()
full_data.PoolQC.fillna('None', inplace=True)

full_data.MiscFeature.fillna('None', inplace=True)

full_data.Alley.fillna('None', inplace=True)

full_data.Fence.fillna('None', inplace=True)

full_data.FireplaceQu.fillna('None', inplace=True)

full_data.LotFrontage = full_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

for col in ['GarageFinish','GarageQual','GarageCond', 'GarageType','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual']:

    full_data[col].fillna('None', inplace=True)

for col in ['GarageCars','GarageArea','GarageYrBlt','BsmtHalfBath','BsmtFullBath','TotalBsmtSF','BsmtUnfSF','BsmtFinSF1','BsmtFinSF2']:

    full_data[col].fillna(0, inplace=True)

full_data.MasVnrType.fillna('None', inplace=True)

full_data.MasVnrArea.fillna(0, inplace=True)

full_data.MSZoning.fillna(full_data.MSZoning.mode()[0], inplace=True)

full_data.Functional.fillna('Typ', inplace=True) #fill with most common value

full_data.drop('Utilities',axis=1,inplace=True) #all values are same, except one, which is in train set

full_data.Electrical.fillna('SBrkr', inplace=True)

full_data['Exterior1st'].fillna(full_data['Exterior1st'].mode()[0], inplace=True)

full_data['Exterior2nd'].fillna(full_data['Exterior2nd'].mode()[0], inplace=True)

full_data.KitchenQual.fillna('TA', inplace=True)

full_data.SaleType.fillna('WD', inplace=True)



full_data.isnull().sum().max()
#plot heatmap of correlation coefficients

plt.figure(figsize=(10,8))

sns.heatmap(train.corr(), vmax = 0.8)



print(train.corr(method = 'spearman')['SalePrice'].abs().nlargest(15).keys())
for col in ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', ]:

    full_data[col] = pd.Categorical(full_data[col], categories=['None','Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered=True).codes



for col in ['BsmtFinType1', 'BsmtFinType2']:

    full_data[col] = pd.Categorical(full_data[col], categories=['None','Unf','LwQ','Rec','BLQ','ALQ','GLQ'], ordered=True).codes



full_data['Functional'] = pd.Categorical(full_data['Functional'], categories=['Sal','Sev','Maj2','Maj1','Mod','Min2','Min1','Typ'], ordered=True).codes

full_data['Fence'] = pd.Categorical(full_data['Fence'], categories=['None','MnWw','GdWo','MnPrv','GdPrv'], ordered=True).codes

full_data['BsmtExposure'] = pd.Categorical(full_data['BsmtExposure'], categories=['None','No','Mn','Av','Gd'], ordered=True).codes

full_data['GarageFinish'] = pd.Categorical(full_data['GarageFinish'], categories=['None','Unf','RFn','Fin'], ordered=True).codes

full_data['LandSlope'] = pd.Categorical(full_data['LandSlope'], categories=['Sev','Mod','Gtl'], ordered=True).codes

full_data['LotShape'] = pd.Categorical(full_data['LotShape'], categories=['IR3','IR2','IR1','Reg'], ordered=True).codes

full_data['PavedDrive'] = pd.Categorical(full_data['PavedDrive'], categories=['N','P','Y'], ordered=True).codes





from sklearn.preprocessing import LabelEncoder



lbl = LabelEncoder()

for col in ['Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond']:

    full_data[col] = lbl.fit_transform(full_data[col])
full_data['TotalSF'] = full_data.TotalBsmtSF + full_data['1stFlrSF'] + full_data['2ndFlrSF']

full_data['YearsSinceRemodel'] = full_data.YrSold.astype(int) - full_data.YearRemodAdd.astype(int)

full_data['TotalHomeQuality'] = full_data.OverallQual + full_data.OverallCond
#select numeric features

numeric_feats = full_data.loc[:,full_data.dtypes != object].columns 

#select features with high skew

skewed_feats = full_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewed_feats = skewed_feats[skewed_feats > 0.75]



#log(1+x) applied to all columns

for col in skewed_feats.keys().to_list():

    full_data[col] = np.log1p(full_data[col])
def addCrossSquared(temp, plist):

    m = temp.shape[1]

    for i in range(len(plist)-1):

        for j in range(i+1,len(plist)):

            temp = temp.assign(newcol=pd.Series(temp[plist[i]]*temp[plist[j]]).values)   

            temp.columns.values[m] = plist[i] + '*' + plist[j]

            m += 1

    return temp
poly_features_list = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF','1stFlrSF']

full_data = addCrossSquared(full_data, poly_features_list)
full_data = pd.get_dummies(full_data)

full_data.shape
full_data['MSZoning_C (all)'].value_counts()

full_data.drop('MSZoning_C (all)', axis=1, inplace=True)
train = full_data[:ntrain]

test = full_data[ntrain:]
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, Ridge, Lasso
scaler = RobustScaler() 

X_train = scaler.fit_transform(train)

X_test = scaler.transform(test)
def rmse_cv(model):

    rmse = np.sqrt(-cross_val_score(model, X_train, y ,scoring = 'neg_mean_squared_error', cv=5))

    return rmse
ridge = Ridge()

alphas = [0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100,200,300,500,1000]

error_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]

sns.lineplot(x = alphas, y = error_ridge, marker = 'o')

plt.xlabel('alpha')

plt.ylabel('RMSE')

plt.xlim(0,100)
ridgecv = RidgeCV(cv = 5).fit(X_train,y)

rmse_cv(ridgecv).mean()
lassocv = LassoCV(cv=5).fit(X_train,y)

print("Number of values searched through: ", lassocv.alphas_.shape[0])

print("Selected alpha: ", lassocv.alpha_)

print("Smallest alpha tried: ", lassocv.alphas_[-1])
alphas = np.linspace(0.00001, 0.055, 50)

lassocv = LassoCV(cv = 5, alphas = alphas).fit(X_train,y)

print("Value of alpha selected: ", lassocv.alpha_)

rmse_cv(lassocv).mean()
coeffs = pd.Series(lassocv.coef_, index = train.columns)

imp_coeffs = pd.concat([coeffs.sort_values().head(10),coeffs.sort_values().tail(10)])

sns.barplot(x = imp_coeffs.values, y = imp_coeffs.keys())

plt.xlabel('Coefficient')

plt.ylabel('Feature')
alphas = np.linspace(0.002, 0.0002, 50)

elasticcv = ElasticNetCV(cv = 5, l1_ratio = 0.7, alphas = alphas).fit(X_train,y)

rmse_cv(elasticcv).mean()
import xgboost as xgb
'''brute_xgb = xgb.XGBRegressor()



max_depth = [3,4,5,6,8,10]

min_child_weight = [3,4,5,6,7,9]

gamma = [i/10.0 for i in range(0,2)]

subsample = [i/100.0 for i in range(60,90)]

colsample_bytree = [i/100.0 for i in range(60,90)]

base_score = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6]

learning_rate = [0.01,0.05,0.1,0.2,0.3,0.5]

n_estimators = [200, 300, 400, 500, 600, 700, 800 ,900, 1000, 1200, 1400]



params = {

    'max_depth': max_depth,

    'gamma': gamma,

    'subsample': subsample,

    'colsample_bytree': colsample_bytree,

    'min_child_weight': min_child_weight,

    'base_score': base_score,

    'learning_rate': learning_rate,

    'n_estimators': n_estimators

}



search = RandomizedSearchCV(estimator = brute_xgb,

                                   param_distributions = params, 

                                   cv = 5,

                                   n_iter = 300,

                                   scoring = 'neg_mean_squared_error',

                                   n_jobs = -1,

                                   verbose = 5,

                                   random_state = 23 )



search.fit(X_train,y)

'''

#done, result copied and re-initialised. no need to do again.
#search.best_estimator_
best_xgb = xgb.XGBRegressor(base_score=0.45, booster=None, colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.62, gamma=0.0, gpu_id=-1,

             importance_type='gain', interaction_constraints=None,

             learning_rate=0.01, max_delta_step=0, max_depth=4,

             min_child_weight=3, monotone_constraints=None,

             n_estimators=3000, n_jobs=0, num_parallel_tree=1,

             objective='reg:squarederror', random_state=0, reg_alpha=0,

             reg_lambda=1, scale_pos_weight=1, subsample=0.61, tree_method=None,

             validate_parameters=False, verbosity=None).fit(X_train,y)
from sklearn.ensemble import VotingRegressor



voting = VotingRegressor([('Lasso',lassocv), ('Ridge',ridgecv), ('ElasticNet', elasticcv), ('XGBoost', best_xgb)])

voting = voting.fit(X_train,y)



rmse_cv(voting).mean()
model = voting

y_pred = model.predict(X_test)

y_pred = np.exp(y_pred)

output = pd.DataFrame({'Id': testID, 'SalePrice': y_pred})

output.to_csv('submission.csv',index=False)