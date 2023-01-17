import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



print("Shape of train: ", train.shape)

print("Shape of test: ", test.shape)
# making copies of train and test



#Save the 'Id' column

train_ID = train['Id']

test_ID = test['Id']

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)
fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()



#Deleting outliers

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)



#Check the graphic again

fig, ax = plt.subplots()

ax.scatter(train['GrLivArea'], train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()

# target variable



from scipy import stats

from scipy.stats import norm



sns.distplot(train['SalePrice'] , fit = norm)



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

print('mu = {:.2f} and sigma = {:.2f}'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot = plt)

plt.show()
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

train["SalePrice"] = np.log1p(train["SalePrice"])



#Check the new distribution 

sns.distplot(train['SalePrice'] , fit = norm)



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

print('mu = {:.2f} and sigma = {:.2f}'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot = plt)

plt.show()
# combining the train and test datasets for preprocessing



ntrain = train.shape[0]

ntest = test.shape[0]



# creating y-train

y_train = train.SalePrice.values



combine = pd.concat([train, test])

combine.drop(['SalePrice'], axis = 1, inplace =  True)



# printing the shape of new dataset

combine.shape
combine_na = (combine.isnull().sum() / len(combine)) * 100

combine_na = combine_na.drop(combine_na[combine_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :combine_na})

missing_data.head(20)

# checking is there are any NULL values in the train and test sets



combine.isnull().sum()
# ## filling the missing values in the Column Types of BsmtFinSF2



# simply filling the NULL value with none

combine['BsmtFinSF2'].fillna(0, inplace = True)



# checking if there are any Null values left

combine['BsmtFinSF2'].isnull().any()
# ## filling the missing values in the Column Types of BsmtFinSF1



# simply filling the NULL value with none

combine['BsmtFinSF1'].fillna(0, inplace = True)



# checking if there are any Null values left

combine['BsmtFinSF1'].isnull().any()
# ## filling the missing values in the Column Types of BsmtFinType2



# simply filling the NULL value with none

combine['BsmtFinType2'].fillna('None', inplace = True)



# checking if there are any Null values left

combine['BsmtFinType2'].isnull().any()
# ## filling the missing values in the Column Types of BsmtFinType1



# simply filling the NULL value with none

combine['BsmtFinType1'].fillna('None', inplace = True)



# checking if there are any Null values left

combine['BsmtFinType1'].isnull().any()
# ## filling the missing values in the Column Types of BsmtFullBath



# simply filling the NULL value with 0 as it is the most common

combine['BsmtFullBath'].fillna(0, inplace = True)



# checking if there are any Null values left

combine['BsmtFullBath'].isnull().any()
# ## filling the missing values in the Column Types of BsmtHalfBath



# simply filling the NULL value with 0 as it is the most common

combine['BsmtHalfBath'].fillna(0, inplace = True)



# checking if there are any Null values left

combine['BsmtHalfBath'].isnull().any()
# ## filling the missing values in the Column Types of BsmtQual



# simply filling the NULL value with none

combine['BsmtQual'].fillna('None', inplace = True)



# checking if there are any Null values left

combine['BsmtQual'].isnull().any()
# ## filling the missing values in the Column Types of BsmtUnfSF



# simply filling the NULL value with 0 as it is the most common

combine['BsmtUnfSF'].fillna(0, inplace = True)



# checking if there are any Null values left

combine['BsmtUnfSF'].isnull().any()
## filling the missing values in the Column Types of Electrical



# simply filling the NULL value with VinylSd as it is the most common

combine['Electrical'].fillna(combine['Electrical'].mode()[0], inplace = True)



# checking if there are any Null values left

combine['Electrical'].isnull().any()
## filling the missing values in the Column Types of Exterior2nd



# simply filling the NULL value with VinylSd as it is the most common

combine['Exterior1st'].fillna(combine['Exterior1st'].mode()[0], inplace = True)



# checking if there are any Null values left

combine['Exterior1st'].isnull().any()
## filling the missing values in the Column Types of Exterior2nd



# simply filling the NULL value with most common value

combine['Exterior2nd'].fillna(combine['Exterior2nd'].mode()[0], inplace = True)



# checking if there are any Null values left

combine['Exterior2nd'].isnull().any()
## filling the missing values in the Column Types of Fence



# simply filling the NULL value with none

combine['Fence'].fillna('None', inplace = True)



# checking if there are any Null values left

combine['Fence'].isnull().any()
## filling the missing values in the Column Types of FireplaceQu



# simply filling the NULL value with none

combine['FireplaceQu'].fillna('None', inplace = True)



# checking if there are any Null values left

combine['FireplaceQu'].isnull().any()
## filling the missing values in the Column Types of MSZoning



# simply filling the NULL value with none

combine['MSZoning'].fillna('None', inplace = True)



# checking if there are any Null values left

combine['MSZoning'].isnull().any()
## filling the missing values in the Column Types of MasVnrArea



# simply filling the NULL value with 0

combine['MasVnrArea'].fillna(0, inplace = True)



# checking if there are any Null values left

combine['MasVnrArea'].isnull().any()
## filling the missing values in the Column Types of MasVnrType



# simply filling the NULL value with none

combine['MasVnrType'].fillna('None', inplace = True)



# checking if there are any Null values left

combine['MasVnrType'].isnull().any()
## filling the missing values in the Column Types of MiscFeature



# simply filling the NULL value with none

combine['MiscFeature'].fillna('None', inplace = True)



# checking if there are any Null values left

combine['MiscFeature'].isnull().any()
## filling the missing values in the Column Typesof PoolQC



# simply filling the NULL value with Ex as it is the most common

combine['PoolQC'].fillna('None', inplace = True)



# checking if there are any Null values left

combine['PoolQC'].isnull().any()
## filling the missing values in the Column SaleType



# simply filling the NULL value with WD as it is the most common

combine['SaleType'].fillna(combine['SaleType'].mode()[0], inplace = True)



# checking if there are any Null values left

combine['SaleType'].isnull().any()
# filling the missing values in the Column TotalBsmtSF



combine['TotalBsmtSF'].fillna(combine['TotalBsmtSF'].mean(), inplace = True)



# checking if there are any Null values left

combine['TotalBsmtSF'].isnull().any()
# checking the unique value in the column Utlities



combine['Utilities'].value_counts()
# AS, we just saw that almost all the rows have same value for Utilities we will get rid of this column



combine.drop(['Utilities'], axis = 1, inplace = True)



# checking the new shape of the dataset

combine.shape
# filling the missing values in the LotFrontage column



#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

combine["LotFrontage"] = combine.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



# checking if there are any NULL values left in the LotFronage Column

combine['LotFrontage'].isnull().any()

# filling the missing values 



# we will replace null values with none

combine['Alley'].fillna('None', inplace = True)



# checking if there are any NULL values left

combine['Alley'].isnull().any()
# filling the missing values in the BsmtCond column



# we are simply filling none in the place NULL values 

combine['BsmtCond'].fillna('None', inplace = True)



# checking if there are any left NULL values

combine['BsmtCond'].isnull().any()
# filling the missing values in the BsmtCond column



# replacing No with None

combine['BsmtExposure'].replace(('No'), ('None'), inplace = True)



# we are simply filling None in the place NULL values 

combine['BsmtExposure'].fillna('None', inplace = True)



# checking if there are any left NULL values

combine['BsmtExposure'].isnull().any()
combine['KitchenQual'].value_counts(dropna = False)
# filling the missing values in the KitchenQual column



# we are simply filling TA in the place NULL values 

combine['KitchenQual'].fillna(combine['KitchenQual'].mode()[0], inplace = True)



# checking if there are any left NULL values

combine['KitchenQual'].isnull().any()
# filling the missing values in the GarageYrBlt column



# we are simply filling none in place of NULL values

combine['GarageYrBlt'].fillna('None', inplace = True)



# checking if there are any left NULL values

combine['GarageYrBlt'].isnull().any()
# filling the missing values in the GarageType column



# we are simply filling none in the place NULL values 

combine['GarageType'].fillna('None', inplace = True)



# checking if there are any left NULL values

combine['GarageType'].isnull().any()
# filling the missing values in the GarageQual column



# we are simply filling none in the place NULL values 

combine['GarageQual'].fillna('None', inplace = True)



# checking if there are any left NULL values

combine['GarageQual'].isnull().any()
# filling the missing values in the GarageFinish column



# we are simply filling none in the place NULL values  

combine['GarageFinish'].fillna('None', inplace = True)



# checking if there are any left NULL values

combine['GarageFinish'].isnull().any()
# filling the missing values in the GarageCond column



# we are simply filling Unf in the place NULL values 

combine['GarageCond'].fillna('None', inplace = True)



# checking if there are any left NULL values

combine['GarageCond'].isnull().any()
# filling the missing values in the GarageCars column



# we are simply filling 0 in the place NULL values 

combine['GarageCars'].fillna(0, inplace = True)



# checking if there are any left NULL values

combine['GarageCars'].isnull().any()
# filling the missing values in the GarageArea column



# we are simply filling 0 in the place NULL values 

combine['GarageArea'].fillna(0, inplace = True)



# checking if there are any left NULL values

combine['GarageArea'].isnull().any()
# filling the missing values in the Functional column



combine['Functional'].fillna(combine['Functional'].mode()[0], inplace = True)



# checking if there are any left NULL values

combine['Functional'].isnull().any()
combine.isnull().sum().sum()
# Transforming some numerical variables that are really categorical



#MSSubClass=The building class

combine['MSSubClass'] = combine['MSSubClass'].apply(str)





#Changing OverallCond into a categorical variable

combine['OverallCond'] = combine['OverallCond'].astype(str)





#Year and month sold are transformed into categorical features.

combine['YrSold'] = combine['YrSold'].astype(str)

combine['MoSold'] = combine['MoSold'].astype(str)
from sklearn.preprocessing import LabelEncoder



cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')



# process columns, apply LabelEncoder to categorical features

for c in cols:

    lb = LabelEncoder() 

    lb.fit(list(combine[c].values)) 

    combine[c] = lb.transform(list(combine[c].values))



# shape        

print('Shape all_data: {}'.format(combine.shape))

# FEATURE ENGINEERING

# adding a new column total area as it is big determinant for prices of a home.



combine['total_area'] = combine['1stFlrSF'] + combine['2ndFlrSF'] + combine['TotalBsmtSF']



# looking at the new shape of the combine dataset

combine.shape

# finding skewed features



from scipy.stats import skew



numerical_feats = combine.dtypes[combine.dtypes != 'object'].index



# checking the skewness in all the numerical features

skewed_feats = combine[numerical_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending = False)



# converting the features into a dataframe

skewness = pd.DataFrame({'skew':skewed_feats})



# checking the head of skewness dataset

skewness.head(10)
# applying box-cox transformations



skewness = skewness[abs(skewness > 0.8)]



# printing how many features are to be box-cox transformed

print("There are {} skewed numerical features to box cox transform".format(skewness.shape[0]))



# importing box-cox1p

from scipy.special import boxcox1p



# defining skewed features

skewed_features = skewness.index



lam = 0.15

for feat in skewed_features:

    combine[feat] += 1

    combine[feat] = boxcox1p(combine[feat], lam)

  

combine[skewed_features] = np.log1p(combine[skewed_features])
# one hot encoding for all the categorical variables



combine = pd.get_dummies(combine)



# checking the head of the dataset

combine.head()
# separating the train and test datasets



x_train = combine.iloc[:ntrain]

x_test = combine.iloc[ntrain:]



# checking the shapes of train and test datasets

print("Shape of train :", x_train.shape)

print("Shape of test :", x_test.shape)
#Validation function

n_folds = 5

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_train.values)

    rmse= np.sqrt(-cross_val_score(model, x_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
# LASSO MODEL

# WITH PIPELINE  and using robust scalerTO AVOID SENSITIVITY TOWARDS OUTLIERS



from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler



from sklearn.linear_model import Lasso



lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.0005, random_state = 3))

lasso.fit(x_train, y_train)

score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# making an Elastic Net model

from sklearn.linear_model import ElasticNet



ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))



score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



ENet.fit(x_train, y_train)

from sklearn.ensemble import GradientBoostingRegressor



# making a gradint boosting model

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)



score = rmsle_cv(GBoost)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# making predictions

GBoost.fit(x_train, y_train)

predictions = GBoost.predict(x_test)
# light gradient boosting

import lightgbm as lgb



model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



model_lgb.fit(x_train, y_train)
predictions = model_lgb.fit(x_train, y_train)
# KERNEL RIDGE REGRESSION



from sklearn.kernel_ridge import KernelRidge



KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

score = rmsle_cv(KRR)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# STACKING

# Simplest model -> Averaging Base Models



from sklearn.base import BaseEstimator

from sklearn.base import RegressorMixin

from sklearn.base import TransformerMixin

from sklearn.base import clone



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

        predictions = np.column_stack([model.predict(X) for model in self.models_])

        return np.mean(predictions, axis=1)   
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))



score = rmsle_cv(averaged_models)

print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

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
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),

                                                 meta_model = lasso)



# score = rmsle_cv(stacked_averaged_models)

# print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
stacked_averaged_models.fit(x_train.values, y_train)

stacked_train_pred = stacked_averaged_models.predict(x_train.values)

stacked_pred = np.expm1(stacked_averaged_models.predict(x_test.values))

# XG BOOST

import xgboost as xgb



model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)



# score = rmsle_cv(model_xgb)

# print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_xgb.fit(x_train, y_train)

xgb_train_pred = model_xgb.predict(x_train)

xgb_pred = np.expm1(model_xgb.predict(x_test))
model_lgb.fit(x_train, y_train)

lgb_train_pred = model_lgb.predict(x_train)

lgb_pred = np.expm1(model_lgb.predict(x_test.values))
# '''RMSE on the entire Train data when averaging'''



# print('RMSLE score on train data:')

# print(rmsle(y_train,stacked_train_pred*0.70 +

#                xgb_train_pred*0.15 + lgb_train_pred*0.15 ))
predictions = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not



submission = pd.DataFrame({'Id': test_ID,'SalePrice': predictions})



#Visualize the first 5 rows

submission.head()
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'submission.csv'



submission.to_csv(filename, index=False)



print('Saved file: ' + filename)