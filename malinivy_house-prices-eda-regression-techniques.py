# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

%matplotlib inline
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

print('Training set ',train_df.shape)

print('test set ',test_df.shape)
train_df.dtypes
train_df.head(5)
test_df.head(5)
#Save the 'Id' column in the test data for submissions

test_ID = test_df['Id']
from scipy.stats import norm

#sns.kdeplot(train_df['SalePrice'],shade=True,fit=norm) # Using log and sqrt transformationsto improve the distrubution

sns.distplot(train_df['SalePrice'],fit=norm)


sns.distplot(np.log1p(train_df["SalePrice"]),fit=norm)



sns.distplot(np.sqrt(train_df["SalePrice"]),fit=norm)

#log Transformation of Target variable

train_df["SalePrice"] = np.log1p(train_df["SalePrice"])

#train_df["SalePrice"] = np.sqrt(train_df["SalePrice"])

y_train = train_df['SalePrice']

df = pd.concat((train_df, test_df)).reset_index(drop=True)

df.drop(['SalePrice'], axis=1, inplace=True)

print("entire dataset size is ",df.shape)
df.columns
df.isnull().sum()[df.isnull().sum() > 0]
cat_df=df.select_dtypes(include=["object"])# only categorical vars
cat_df.dtypes
cat_df.shape
cat_df.isnull().sum()[cat_df.isnull().sum() > 0]
for i in ("MasVnrType",'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 

            'BsmtFinType2','Alley','Fence','FireplaceQu'):

    print(cat_df[i].value_counts())
#cat_df[cat_df["MasVnrType"].isnull()]
for col in ("MasVnrType",'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 

            'BsmtFinType2','MSSubClass','Alley','Fence','FireplaceQu'):

    df[col] = df[col].fillna('None')
for col in ("MSZoning",'Electrical', 'KitchenQual', 'Exterior1st', 'KitchenQual',

            'Exterior2nd', 'SaleType','Functional','Electrical'):

    print(cat_df[i].value_counts())
#using the most commonly occuring category to fill na 

for col in ("MSZoning",'Electrical', 'KitchenQual', 'Exterior1st', 'KitchenQual',

            'Exterior2nd', 'SaleType','Functional','Electrical'):

    df[col] = df[col].fillna(df[col].mode()[0])


# apply LabelEncoder to categorical features to use the rank and order

from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')



for i in cols:

    labelenc = LabelEncoder() 

    labelenc.fit(list(df[i].values)) 

    df[i] = labelenc.transform(list(df[i].values))



#using one hot encoding 

df = pd.get_dummies(df)

print(df.shape)
numDf=df._get_numeric_data()

numDf.columns
numDf.isnull().sum()[numDf.isnull().sum() > 0]
columns = ["MasVnrArea", "BsmtUnfSF", 'BsmtFullBath', 'BsmtHalfBath',"TotalBsmtSF", "BsmtFinSF2", "BsmtFinSF1", "GarageArea",'GarageYrBlt', 'GarageCars']

for col in columns:

    df[col].fillna(0,inplace= True)
df["LotFrontage"] = df['LotFrontage'].fillna(df[col].median())
df.isnull().sum()[df.isnull().sum() > 0]
df.ix[1:15,['1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea']]
#grLivArea seems to be a sum of the rest of the area ,1stSF,2ndFlrSF,LowQualFinSF could be dropped

#df= df.drop(['1stFlrSF','2ndFlrSF','LowQualFinSF'],axis=1)
# Adding  features

#df['Porch_area']=df['OpenPorchSF']+df['EnclosedPorch']+df['3SsnPorch']+df['ScreenPorch']



df['TotalSqFt'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

#Age of the house

#df['houseAge']=0

#df['houseAge'] = df['YrSold'] - df['YearBuilt']

#whether or not it was remodelled

df['renovated']=0

df.loc[df['YearRemodAdd'] > 0, 'renovated'] = 1
#sns.distplot(df['houseAge'])

#res = stats.probplot(df['houseAge'], plot=plt)

#df['YearBuilt'].head(10)
#df.isnull().sum()[df.isnull().sum() > 0]

df.isnull().sum()
#checking for skewness

from scipy.stats import skew



newDf_numeric=df.select_dtypes(exclude=["object"])

skewness =df.apply(lambda x: skew(x))

skewness_features = skewness[abs(skewness) >= 0.75 ].index



#df[skewness_features] = np.log1p(df[skewness_features])
from scipy.special import boxcox1p

for feat in skewness.index:

    df[feat] = boxcox1p(df[feat], 0.15)

    
from sklearn.linear_model import ElasticNet, Lasso

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb
train = df[:1460]

test = df[1460:]
#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
from sklearn.ensemble import RandomForestRegressor

RandomForestRegressor = RandomForestRegressor(n_estimators = 200, random_state=0,

                                  max_features=7,max_depth=10,min_samples_leaf =3,oob_score = 0.8)



RandomForestRegressor.fit(train, y_train)

print('RandomForestRegressor',rmsle_cv(RandomForestRegressor).mean())
# can further improve by using only the coefficients of importance

x=train.columns

coef = pd.Series(RandomForestRegressor.feature_importances_, index = train.columns)

imp_coef = pd.concat([coef.sort_values().head(20),

                     coef.sort_values().tail(20)])

#y=np.sort(RandomForestRegressor.feature_importances_)

import matplotlib

#Plot Coefficients

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Random forest Model")

plt.barh(x, y)

plt.show()
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)



GBoost = GradientBoostingRegressor(n_estimators=3000,

                                   learning_rate=0.05,

                                   max_depth=4,

                                   min_samples_leaf=15,

                                   min_samples_split=10, 

                                   loss='huber')





model_xgb = xgb.XGBRegressor(gamma=0.0468, 

                             learning_rate=0.05, 

                             max_depth=3, 

                              n_estimators=2200)
score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(KRR)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_xgb)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds

   # We again fit the data on clones of the original models

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        

        # Train cloned base models then create out-of-fold predictions

        # that are needed to train the cloned meta-model

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(X[train_index], y[train_index])

                y_pred = instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

                

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

    #Do the predictions of all base models on the test data and use the averaged predictions as 

    #meta-features for the final prediction which is done by the meta-model

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),

                                                 meta_model = lasso)



score = rmsle_cv(stacked_averaged_models)

print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))



stacked_averaged_models.fit(train.values, y_train)

stacked_train_pred = stacked_averaged_models.predict(train.values)

stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))

print(rmsle(y_train, stacked_train_pred))
model_xgb.fit(train, y_train)

xgb_train_pred = model_xgb.predict(train)

xgb_pred = np.expm1(model_xgb.predict(test))

print(rmsle(y_train, xgb_train_pred))
print('RMSLE score on train data:')

print(rmsle(y_train,stacked_train_pred*0.85 +

               xgb_train_pred*0.15 ))
ensemble = stacked_pred*0.85 + xgb_pred*0.15 
sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = ensemble

sub.to_csv('submission.csv',index=False)