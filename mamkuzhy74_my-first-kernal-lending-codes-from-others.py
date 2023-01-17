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
import random as rnd



from datetime import tzinfo, timedelta, datetime



from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.metrics import mean_squared_error, r2_score



from sklearn.linear_model import LinearRegression, ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold, cross_val_score, train_test_split,GridSearchCV

from sklearn.tree import export_graphviz

from sklearn.svm import SVC

import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler

from matplotlib import pyplot as plt



from scipy import stats

from scipy.stats import norm, skew #for some statistics



from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

import xgboost as xgb

#import lightgbm as lgb



import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)



%matplotlib inline
#read the csv files in to data frame

df_train = pd.read_csv("../input/train.csv",sep=",")

df_test = pd.read_csv("../input/test.csv",sep=",")
fig, ax = plt.subplots()

ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
#Deleting outliers

df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)
fig, ax = plt.subplots()

ax.scatter(x = df_train['LotArea'], y = df_train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('LotArea', fontsize=13)

plt.show()
df_train = df_train.drop(df_train[(df_train['LotArea']>100000) & (df_train['SalePrice']<400000)].index)
#Split and save fields for later use

dy_train = df_train['SalePrice']

id_test = df_test['Id']

#Combine the dataframe for easy process

df_merge = pd.concat([df_train,df_test],sort=False)

df_merge = df_merge.drop(['SalePrice'],axis=1)
#Check for null values

df_merge.isnull().sum()[df_merge.isnull().sum() > 0]
#Fill the blanks

df_merge['MSZoning']=df_merge['MSZoning'].fillna(df_merge['MSZoning'].mode()[0])

df_merge['LotFrontage']=df_merge['LotFrontage'].fillna(df_merge['LotFrontage'].mean())

df_merge['Alley']=df_merge['Alley'].fillna('None')

df_merge['Utilities']=df_merge['Utilities'].fillna(df_merge['Utilities'].mode()[0])

df_merge['Exterior1st']=df_merge['Exterior1st'].fillna(df_merge['Exterior1st'].mode()[0])

df_merge['Exterior2nd']=df_merge['Exterior2nd'].fillna(df_merge['Exterior2nd'].mode()[0])

df_merge['MasVnrType']=df_merge['MasVnrType'].fillna('None')

df_merge['MasVnrArea']=df_merge['MasVnrArea'].fillna(0)

df_merge['BsmtQual']=df_merge['BsmtQual'].fillna('None')

df_merge['BsmtCond']=df_merge['BsmtCond'].fillna('None')

df_merge['BsmtExposure']=df_merge['BsmtExposure'].fillna('None')

df_merge['BsmtFinType1']=df_merge['BsmtFinType1'].fillna('None')

df_merge['BsmtFinType2']=df_merge['BsmtFinType2'].fillna('None')

df_merge['BsmtFinSF1']=df_merge['BsmtFinSF1'].fillna(0)

df_merge['BsmtFinSF2']=df_merge['BsmtFinSF2'].fillna(0)

df_merge['BsmtUnfSF']=df_merge['BsmtUnfSF'].fillna(0)

df_merge['TotalBsmtSF']=df_merge['TotalBsmtSF'].fillna(0)

df_merge['Electrical']=df_merge['Electrical'].fillna(df_merge['Electrical'].mode()[0])

df_merge['BsmtFullBath']=df_merge['BsmtFullBath'].fillna(0)

df_merge['BsmtHalfBath']=df_merge['BsmtHalfBath'].fillna(0)

df_merge['KitchenQual']=df_merge['KitchenQual'].fillna(df_merge['KitchenQual'].mode()[0])

df_merge['Functional']=df_merge['Functional'].fillna(df_merge['Functional'].mode()[0])

df_merge['FireplaceQu']=df_merge['FireplaceQu'].fillna('None')

df_merge['GarageType']=df_merge['GarageType'].fillna('None')

df_merge['GarageYrBlt']=df_merge['GarageYrBlt'].fillna(1800)

df_merge['GarageFinish']=df_merge['GarageFinish'].fillna('None')

df_merge['GarageCars']=df_merge['GarageCars'].fillna(0)

df_merge['GarageArea']=df_merge['GarageArea'].fillna(0)

df_merge['GarageQual']=df_merge['GarageQual'].fillna('None')

df_merge['GarageCond']=df_merge['GarageCond'].fillna('None')

df_merge['PoolQC']=df_merge['PoolQC'].fillna('None')

df_merge['Fence']=df_merge['Fence'].fillna('None')

df_merge['MiscFeature']=df_merge['MiscFeature'].fillna('None')

df_merge['SaleType']=df_merge['SaleType'].fillna('WD')
#Check for null values agan to confirm

df_merge.isnull().sum()[df_merge.isnull().sum() > 0]
#Build new features 

df_merge["BldAge"] = 2012 - df_merge["YearBuilt"]

#df_merge.loc[(df_merge["YearBuilt"] == df_merge["YearRemodAdd"]),'BldOrig'] = 1

df_merge["BldOrig"]= df_merge["BldOrig"].fillna(0)

df_merge = df_merge.drop(['YearBuilt','YearRemodAdd'], axis = 1)



df_merge["GrgAge"] = 2012 - df_merge["GarageYrBlt"]

df_merge.loc[(df_merge["GarageYrBlt"] == 1800),'GrgAge'] = 0

df_merge = df_merge.drop(['GarageYrBlt'], axis = 1)

df_merge["TotalCarpetArea"] = df_merge["TotalBsmtSF"]+df_merge["GrLivArea"]

df_merge["TotalExtraArea"] = df_merge["WoodDeckSF"]+df_merge["OpenPorchSF"]+df_merge["EnclosedPorch"]+df_merge["3SsnPorch"]+df_merge["ScreenPorch"]

df_merge["CondQualWt"] = df_merge["OverallCond"]*df_merge["OverallQual"]

df_merge["BathCount"] = df_merge["BsmtFullBath"]+df_merge["BsmtHalfBath"]+df_merge["FullBath"]+df_merge["HalfBath"]



df_merge["YrSold"] = (2012 - df_merge["YrSold"]) #+ (12-df_merge["MoSold"])/12

#df_merge = df_merge.drop(['MoSold'], axis = 1)

df_merge["TotalCarpetArea"] = df_merge["TotalCarpetArea"].map(lambda i: np.log(i) if i > 0 else 0)

df_merge["TotalExtraArea"] = df_merge["TotalExtraArea"].map(lambda i: np.log(i) if i > 0 else 0)
#Drop some fields not useful

df_merge = df_merge.drop(['WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'], axis = 1)

df_merge = df_merge.drop(['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF'], axis = 1)

df_merge = df_merge.drop(['1stFlrSF','2ndFlrSF','LowQualFinSF'], axis = 1)
#Set properties ofsome features

df_merge["MSSubClass"] = df_merge["MSSubClass"].astype(str)

df_merge["OverallCond"] = df_merge["OverallCond"].astype(str)

df_merge["OverallQual"] = df_merge["OverallQual"].astype(str)

#df_merge["YearBuilt"] = df_merge["YearBuilt"].astype(str)

#df_merge["YearRemodAdd"] = df_merge["YearRemodAdd"].astype(str)

#df_merge["GarageYrBlt"] = df_merge["GarageYrBlt"].astype(str)

df_merge["YrSold"] = df_merge["YrSold"].astype(str)

df_merge["MoSold"] = df_merge["MoSold"].astype(str)

#Encoding the features 

df_cat = df_merge.select_dtypes(include=['object'])

df_num = df_merge.select_dtypes(exclude=['object'])

#apply label encoder for categorical columns 

le = LabelEncoder()

df_cat = df_cat.apply(le.fit_transform)

#perform one hot encoding on categorical columns 

#df_cat_dum = pd.get_dummies(df_cat.astype(str))

#Concatenate dummy columns, numeric and the target (dependent variable) as a final dataframe

#df_merge = pd.concat([df_num,df_cat_dum],axis=1)

df_merge = pd.concat([df_num,df_cat],axis=1)
#split the processed dataframe 

df_merge = df_merge.drop(['Id'],axis=1)

df_train = df_merge.iloc[:len(df_train), :]

df_test  = df_merge.iloc[len(df_train):, :]
Xtrain,Xtest,ytrain,ytest = train_test_split(df_train,dy_train,test_size=0.2,random_state=42)
def score_model(model):

    model.fit(Xtrain,ytrain)

    ypred = model.predict(Xtest)

    return r2_score(ytest,ypred)



def pred_model(model):

    model.fit(df_train,dy_train)

    ypred = model.predict(df_test)

    return ypred
RanFo = RandomForestRegressor(n_estimators=120)

score_model(RanFo)
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

score_model(ENet)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

score_model(GBoost)
XGBoost = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)

score_model(GBoost)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

score_model(lasso)
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
averaged_models = AveragingModels(models = (RanFo,GBoost,XGBoost))

score_model(averaged_models)
averaged_models.fit(df_train,dy_train)
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

                #instance.fit(np.ndarray(X)[train_index], np.ndarray(y)[train_index])

                instance.fit(X.iloc[train_index], y.iloc[train_index])

                y_pred = instance.predict(X.iloc[holdout_index])

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
#stacked models for prediction

models = (RanFo,GBoost,XGBoost)

stacked_averaged_models = StackingAveragedModels(base_models = models,

                                                 meta_model = lasso)

score_model(stacked_averaged_models)
stacked_averaged_models.fit(df_train,dy_train)
#predict using stacked models

pred_out = pd.DataFrame({'ID':id_test})

pred_out['SalePrice'] = stacked_averaged_models.predict(df_test)
#Prediction based on average model

pred_df = pd.DataFrame({'ID':id_test})

pred_df['SalePrice'] = averaged_models.predict(df_test)
#Submit the results

pred_df.to_csv('results.csv', index=False)