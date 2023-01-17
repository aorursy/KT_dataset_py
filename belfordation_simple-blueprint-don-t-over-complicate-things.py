import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

import statsmodels.api as sm

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 2000)

pd.set_option('display.width', 1000)
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.corr()['SalePrice'].sort_values(ascending=False).head(10)
sns.scatterplot(train['GrLivArea'],train['SalePrice'])
def find_outliers_tukey(x):

    q1 = np.percentile(x, 25)

    q3 = np.percentile(x, 75)

    iqr = q3-q1   

    floor = q1 - 1.5*iqr

    ceiling = q3 + 1.5*iqr

    outlier_indices = list(x.index[(x < floor)|(x > ceiling)])

    outlier_values = list(x[outlier_indices])



    return outlier_indices

out=find_outliers_tukey(train['TotalBsmtSF'])
train=train.drop(out)
#take another look:

sns.scatterplot(train['GrLivArea'],train['SalePrice'])
sns.distplot((train['SalePrice']))
train['SalePrice']=np.log(train['SalePrice'])

sns.distplot(train['SalePrice'])
dataset = pd.concat(objs=[train, test], axis=0,sort=False,ignore_index=True)



dataset.isnull().sum().sort_values(ascending=False)
from sklearn.impute import SimpleImputer

imp_num=SimpleImputer(missing_values=np.nan,strategy='mean') #mean for numericals and mode for categoricals

dataset[['MasVnrArea','LotFrontage','GarageArea']]=pd.DataFrame(imp_num.fit_transform(dataset[['MasVnrArea','LotFrontage'

                                                                                               ,'GarageArea']]))

imp_cat=SimpleImputer(missing_values=np.nan,strategy='most_frequent')



dataset[['Electrical','MasVnrType','SaleType','MSZoning','Utilities','Exterior1st','Exterior2nd','BsmtFullBath','BsmtHalfBath','KitchenQual','Functional','SaleType']]=pd.DataFrame(imp_cat.fit_transform(dataset[['Electrical','MasVnrType','SaleType','MSZoning','Utilities','Exterior1st','Exterior2nd','BsmtFullBath','BsmtHalfBath','KitchenQual','Functional','SaleType']]))
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):  

    dataset[col] = dataset[col].fillna(0)

    

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    dataset[col] = dataset[col].fillna('Nothing')    
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    dataset[col] = dataset[col].fillna(0)

    

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    dataset[col] = dataset[col].fillna('Nothing')    
dataset['Alley'] = dataset['Alley'].fillna('Nothing')

dataset['FireplaceQu'] = dataset['FireplaceQu'].fillna('Nothing')

dataset['Fence'] = dataset['Fence'].fillna('Nothing')

dataset['PoolQC']=dataset['PoolQC'].fillna('Nothing')

dataset['MiscFeature']=dataset['MiscFeature'].fillna('Nothing')
import pandas as pd

dataset.isnull().sum().sort_values(ascending=False).head()
dataset.dtypes
dataset['MSSubClass']=dataset['MSSubClass'].astype('str')

dataset['MoSold']=dataset['MoSold'].astype('str')

#instead of 2006,2007... label them as 0,1 ... for ease in use

from sklearn.preprocessing import LabelEncoder

cat_encoder=LabelEncoder()

print(cat_encoder.fit_transform(dataset['YrSold'].values))

dataset['YrSold']=cat_encoder.fit_transform(dataset['YrSold'].values)



external=['ExterQual','ExterCond','HeatingQC']

for e in external:

    dataset[e]=dataset[e].map({'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0})

    

    

basement=['BsmtQual','BsmtCond','GarageQual','GarageCond','FireplaceQu','KitchenQual']

for b in basement:

    dataset[b]=dataset[b].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'Nothing':0})

    



dataset['BsmtFinType2']=dataset['BsmtFinType2'].map({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'Nothing':0})

dataset['BsmtFinType1']=dataset['BsmtFinType1'].map({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'Nothing':0})

dataset['BsmtExposure']=dataset['BsmtExposure'].map({'Gd':4,'Av':3,'Mn':2,'No':1,'Nothing':0})

dataset['LandSlope']=dataset['LandSlope'].map({'Gtl':2,'Mod':1,'Sev':0})

dataset['Fence']=dataset['Fence'].map({'GdPrv':4,'MnPrv':3,'GdWo':2,'MnWw':1,'Nothing':0})

    

def add_UltimateYear_ix (X):

    UltimateYear_ix = X[:,YearBuilt]+X[:,YearRemodAdd]+X[:,YrSold]

UltimateYear=pd.DataFrame(data={'UltimateYear':dataset['YearBuilt']+dataset['YearRemodAdd']+dataset['YrSold']})

dataset.insert(loc=60,column='UltimateYear',value=UltimateYear)

dataset=dataset.drop(['YearBuilt','YearRemodAdd','YrSold'],axis=1)
#Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . 

dataset=dataset.drop(['Utilities'],axis=1)

dataset=dataset.drop(['Id'],axis=1) #and drop id column cuase there in no use for it in our model

for col_name in dataset.columns:

    if dataset[col_name].dtypes=='object':

        unique_cat=len(dataset[col_name].unique())

        print("feature {col_name} has {unique_cat} unique categories".format(col_name=col_name,unique_cat=unique_cat)) #intresting syntax!

        
pd.value_counts(dataset['Exterior1st'],normalize=True).sort_values(ascending=False)*100

pd.value_counts(dataset['Exterior2nd'],normalize=True).sort_values(ascending=False)*100
pd.value_counts(dataset['Neighborhood']).sort_values(ascending=False)
pd.value_counts(dataset['MSSubClass']).sort_values(ascending=False)
pd.value_counts(dataset['Condition1']).sort_values(ascending=False)
# In this case, bucket low frequecy categories as "Other"

dataset['Exterior1st']=dataset['Exterior1st'].replace(['ImStucc','CBlock','Stone','AsphShn','BrkComm','Stucco','AsbShng','WdShing','BrkFace'],'other')

dataset['Exterior2nd']=dataset['Exterior2nd'].replace(['ImStucc','CBlock','Stone','AsphShn','Brk Cmn','Stucco','AsbShng','WdShing','Other','BrkFace'],'other')

dataset['Neighborhood']=dataset['Neighborhood'].replace(['Blueste','NPkVill','Veenker','Blmngtn','BrDale','MeadowV','ClearCr'],'other')

dataset['MSSubClass']=dataset['MSSubClass'].replace(['150','40','180','45','75'],'other')

dataset['Condition1']=dataset['Condition1'].replace(['RRNe','RRNn','PosA','RRAe','PosN','RRAn'],'other')

from scipy.stats import norm, skew

numeric_feats = dataset.dtypes[dataset.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = dataset[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness
skewness = skewness[abs(skewness) > 0.75].dropna()

skewness.shape[0]
skewness = skewness[abs(skewness) > 0.75].dropna()

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p,inv_boxcox

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    #all_data[feat] += 1

    dataset[feat] = boxcox1p(dataset[feat], lam)

    

#all_data[skewed_features] = np.log1p(all_data[skewed_features])
train_objs_num = len(train)

dataset_preprocessed = pd.get_dummies(dataset,drop_first=True)

train_preprocessed = dataset_preprocessed[:train_objs_num]

test_preprocessed = dataset_preprocessed[train_objs_num:]
#from sklearn.utils import shuffle

#df = shuffle(df)
X_train=train_preprocessed.drop(['SalePrice'],axis=1)

y_train=train_preprocessed['SalePrice']

Test=test_preprocessed.drop(['SalePrice'],axis=1)
X_train.shape
# Such a large set of features can cause overfitting and also slow computing

# Use feature selection to select the most important features

import sklearn.feature_selection



select = sklearn.feature_selection.SelectKBest(k=180)

selected_features = select.fit(X_train, y_train)

indices_selected = selected_features.get_support(indices=True)

colnames_selected = [X_train.columns[i] for i in indices_selected]



X_train_selected = X_train[colnames_selected]

X_test_selected = Test[colnames_selected]
(colnames_selected)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC,LinearRegression

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb

from sklearn.model_selection import RandomizedSearchCV
def rmsle(model):   #cross validation for 5 fold

    rmse= np.sqrt(-cross_val_score(model, X_train_selected, y_train, scoring="neg_mean_squared_error",cv =5))

    return(rmse.mean())

gb_reg=GradientBoostingRegressor(n_estimators=2000, learning_rate=0.02,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=30, min_samples_split=30, 

                                   loss='huber')

#gb_reg.fit(X_train,y_train);

rmsle(gb_reg)
forest_reg=RandomForestRegressor(n_estimators=200,max_features=14)

forest_reg.fit(X_train,y_train);

rmsle(forest_reg)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

rmsle(lasso)
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=3))

rmsle(ENet)
KRR = KernelRidge(alpha=0.6, kernel='linear', degree=2, coef0=2.5)

rmsle(KRR)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)

rmsle(model_xgb)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=500,

                              max_bin = 90, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

rmsle(model_lgb)
br=BayesianRidge()

rmsle(br)
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # we define clones of the original models to fit the data in

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self

    

    #Now we do the predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1) 
averaged_models = AveragingModels(models = (ENet, gb_reg, KRR, lasso,model_lgb))



rmsle(averaged_models)
#averaged_models.fit(X_train,y_train)

#predictions=(averaged_models.predict(Test))

#predictionsdf = pd.DataFrame({'Predictions':np.exp(predictions)})

#predictionsdf.to_csv(r'C:\Users\Talion\Desktop\predictions.csv')