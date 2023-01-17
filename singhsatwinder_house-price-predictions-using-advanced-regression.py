# importing the necessary files



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

color = sns.color_palette() # Return a list of colors defining a color palette

sns.set_style('darkgrid')





from scipy import stats

from scipy.stats import norm,skew



import os

print(os.listdir("../input"))
train=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train.head()
test.head()
train.info()
# Descriptive statistics

# describing the numerical features

train.describe()
# describing the categorial features

train.describe(include=['O'])
# checking the number of samples and features

print('shape of train data before droping ID {}'.format(train.shape))

print('shape of test data before droping ID {}'.format(test.shape))



# Id coloumn is unecessary for prediction so we drop it

train.drop('Id',axis=1,inplace=True)

test_id=test['Id']

test.drop('Id',axis=1,inplace=True)



# shape of data after dropping ID column

print('\n')

print('shape of train data after droping ID {}'.format(train.shape))

print('shape of test data after droping ID {}'.format(test.shape))
# Relationship with categorical features

plt.figure(figsize=(13,8))

sns.barplot(x=train['OverallQual'], y=train['SalePrice'] )
fig,axes=plt.subplots(figsize=(12,8))

fig=sns.boxplot(x=train['Neighborhood'], y=train['SalePrice'])

plt.xticks(rotation=90)
plt.figure(figsize=(13,8))

plt.xticks(rotation=90)

sns.barplot(x=train['YearBuilt'], y=train['SalePrice'] )
corrmat=train.corr()

plt.figure(figsize=(10,8))

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index #Get the rows of a DataFrame sorted by the n largest values of columns.

cm = np.corrcoef(train[cols].values.T)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True,  yticklabels=cols.values, xticklabels=cols.values)

plt.show()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], height = 2.5)

plt.show()
#outilers



plt.scatter(x = train['LotFrontage'], y = train['SalePrice'])

plt.ylabel('SalePrice',fontsize=13)

plt.xlabel('LotFrontage', fontsize=13)

plt.show()
train[(train['LotFrontage']>300) & (train['SalePrice']<300000)].index
#Deleting outliers

train = train.drop(train[(train['LotFrontage']>300) & (train['SalePrice']<300000)].index)



#Check the graphic again

plt.scatter(x = train['LotFrontage'], y = train['SalePrice'])

plt.ylabel('SalePrice',fontsize=13)

plt.xlabel('LotFrontage', fontsize=13)

plt.show()
# SalesPrice is the variable we need to predict.Let's explore it 

sns.distplot(train['SalePrice'],fit=norm)



mu,sigma=norm.fit(train['SalePrice'])



print('\n mu {:.2f} and sigma {:.2f} is '.format(mu,sigma))



plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu,sigma)], loc='best')



plt.ylabel('Frequency')

plt.title('SalePrice distribution')



plt.show()
# sale price is right-skewed

print(f"Skewness: {train['SalePrice'].skew()}" )
# Log-transformation of the target variable

# we use numpy log1p which applies log(1+x) to all the elements of Coloumn



train['SalePrice']=np.log1p(train['SalePrice'])



sns.distplot(train['SalePrice'],fit=norm)



mu,sigma=norm.fit(train['SalePrice'])

print('\n mu {:.2f} and sigma {:.2f} is '.format(mu,sigma))



plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu,sigma)], loc='best')



plt.ylabel('Frequency')

plt.title('SalePrice distribution')





plt.show()

# concatenate the train and test data in the same dataframe



ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train.SalePrice.values

all_data = pd.concat((train, test),sort=True).reset_index(drop=True)



all_data.drop(['SalePrice'], axis=1, inplace=True)



print("all_data size is : {}".format(all_data.shape))

all_data.head()
# Missing Data

plt.figure(figsize=(20,6))

sns.heatmap(all_data.isnull())
# calculating percentage of missing data



all_data_na=(all_data.isnull().sum() / len(all_data)) * 100 

all_data_na=all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:34]



missing_data=pd.DataFrame(all_data_na,columns=['Missing Perc.'])

missing_data.sort_values(by='Missing Perc.',ascending=False,inplace=True)

missing_data.head(10)
plt.figure(figsize=(15,10))

sns.barplot(x=all_data_na.index,y=missing_data['Missing Perc.'])

plt.xticks(rotation=90)

plt.xlabel('Features',fontsize=15)

plt.ylabel('Missing Percentage',fontsize=15)

plt.title('percentage of missing values')

plt.show()
# correlation of different feature with price

corrmat=train.corr()

plt.figure(figsize=(15,10))

sns.heatmap(corrmat)



plt.show()
all_data['PoolQC'].unique()
all_data['PoolQC'].value_counts()
all_data['PoolQC']=all_data['PoolQC'].fillna('None')
all_data['MiscFeature'].value_counts()
all_data['MiscFeature']=all_data['MiscFeature'].fillna('None')
all_data['Alley'].value_counts()
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data['Fence'].value_counts()
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data['FireplaceQu'].value_counts()
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["LotFrontage"]=all_data.groupby("Neighborhood")['LotFrontage'].apply(lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'].value_counts()
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data['Utilities'].value_counts()
all_data = all_data.drop(['Utilities'], axis=1)
all_data['Functional'].value_counts()
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'].value_counts()
all_data['Electrical'] = all_data['Electrical'].fillna('SBrkr')
all_data['KitchenQual'].value_counts()
all_data['KitchenQual'].isnull().value_counts()
all_data['KitchenQual'] = all_data['KitchenQual'].fillna('TA')
all_data['Exterior1st'].isnull().value_counts()
all_data['Exterior2nd'].isnull().value_counts()
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'].isnull().value_counts()
all_data['SaleType'].value_counts()
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
# checking for any missing value

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head()
# transforming some numerical values that are really categorial

type(all_data['MSSubClass'][0])
# MSSubClass=The building class

all_data['MSSubClass']=all_data['MSSubClass'].apply(str)



# Changing OverallCond into a categorical variable

all_data['OverallCond']=all_data['OverallCond'].astype(str)



# Year and month sold are transformed into categorical features.

all_data['YrSold']=all_data['YrSold'].apply(str)

all_data['MoSold']=all_data['MoSold'].apply(str)
# Applying label encoder to categorial values



from sklearn.preprocessing import LabelEncoder



cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')



for c in cols:

    lbt=LabelEncoder()

    lbt.fit((all_data[c]))

    lbt.transform((all_data[c]))

    

print('Shape all_data: {}'.format(all_data.shape))

    

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
# skewed Features

# or normally distributed data, the skewness should be about 0. a skewness value > 0 means that 

# there is more weight in the left tail of the distribution
# skew function for checking skewness in data

skew(all_data['MiscVal'])
# check the skewness of all numeric features

numeric_feats= all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    #all_data[feat] += 1

    all_data[feat] = boxcox1p(all_data[feat], lam)
# One-Hot Encoding

all_data = pd.get_dummies(all_data)

print(all_data.shape)
# getting new training and testing sets

train = all_data[:ntrain]

test = all_data[ntrain:]

all_data.columns
from sklearn.linear_model import LinearRegression, Lasso , ElasticNet ,Ridge

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge #combines ridge regression (linear least squares with l2-norm regularization) with the kernel trick

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler #Scale features using statistics that are robust to outliers

from sklearn.base import BaseEstimator, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error,mean_squared_error

import xgboost as xgb

import lightgbm as lgb # LightGBM is a gradient boosting framework that uses tree based learning algorithms.faster and high accuracy

from sklearn.model_selection import GridSearchCV
# defining cross validation 

# we used cross_val_score of sklearn however it does not have shuffle attribute So, we write a bit of code for that
# validation function

n_folds=5



def rmsle_cv(model):

    kf=KFold(n_splits=n_folds,shuffle=True,random_state=42).get_n_splits(train.values)

    rmse=np.sqrt(-cross_val_score(model,X=train.values,y=y_train,scoring="neg_mean_squared_error",cv=kf))

    return rmse

    
linear_reg=make_pipeline(RobustScaler(),LinearRegression())
# parameter tunning to find best value of alpha

param_grid={'alpha' : [0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5]}



grid=GridSearchCV(Lasso(),param_grid=param_grid,scoring='neg_mean_squared_error',cv=4)

grid.fit(train.values,y_train)
grid.best_params_
lasso=make_pipeline( RobustScaler() , Lasso(alpha= 0.0005, random_state=1))
Cs = [0.001, 0.01, 0.1, 1, 10]



gammas = [0.001, 0.01, 0.1, 1]



degree=[2,3,4,5]



param_grid = {'C': Cs, 'gamma' : gammas  , 'degree':degree}



grid=GridSearchCV(SVR(),param_grid=param_grid,cv=5)

grid.fit(train.values,y_train) 
grid.best_params_
support_vector_machine=make_pipeline( RobustScaler(), SVR(C=10,degree=2,gamma=0.001))
ENet = make_pipeline( RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9 ,random_state=3))
KRR = KernelRidge(alpha=0.6 , kernel='polynomial', degree=2)
grid_param={'n_estimators':[20,40,100,200,500,1000,3000]}



grid=GridSearchCV(GradientBoostingRegressor(),param_grid=grid_param,cv=4,scoring="neg_mean_squared_error")

grid.fit(train.values,y_train) 
grid.best_params_
# Its a general thumb-rule to start with square root.

GBoost = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,

                                   max_depth=5, max_features='sqrt', random_state=5)       
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603,  n_estimators=2000,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             learning_rate=0.05, max_depth=4,

                             random_state =7)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=7,

                              learning_rate=0.05, n_estimators=720,

                              bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              min_data_in_leaf =6)
score= rmsle_cv(linear_reg)

print(score)

print("\nLinear regression score mean {:.4f} and Std {:.4f}".format(score.mean(),score.std()))
score= rmsle_cv(lasso)

print(score)

print("\nLasso score mean {:.4f} and Std {:.4f}".format(score.mean(),score.std()))
score= rmsle_cv(support_vector_machine)

print(score)

print("\nLasso score mean {:.4f} and Std {:.4f}".format(score.mean(),score.std()))
score= rmsle_cv(ENet)

print(score)

print("\nElasticNet score mean {:.4f} and Std {:.4f}".format(score.mean(),score.std()))
score= rmsle_cv(KRR)

print(score)

print("\nkernalRigdeRegessor score mean {:.4f} and Std {:.4f}".format(score.mean(),score.std()))
score= rmsle_cv(GBoost)

print(score)

print("\nGradientBoosting score mean {:.4f} and Std {:.4f}".format(score.mean(),score.std()))
score= rmsle_cv(model_xgb)

print(score)

print("\nLightweightGradientBoosting score mean {:.4f} and Std {:.4f}".format(score.mean(),score.std()))
score= rmsle_cv(model_lgb)

print(score)

print("\nLightweightGradientBoosting score mean {:.4f} and Std {:.4f}".format(score.mean(),score.std()))
class AveragingModels(BaseEstimator, RegressorMixin):

    def __init__(self,models):

        self.models=models

        

    # we define the clone of the original model to fit the data in

    def fit(self, X, y):

        self.models_=[clone(x) for x in self.models]

        

        # train the base models

        for model in self.models_:

            model.fit(X,y)

            

        return self

    #Now we do the predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([ model.predict(X) for model in self.models_])

        return np.mean(predictions, axis=1)   
# We just average four models here ENet, GBoost, KRR and lasso. but we can add more models in mix.

averaging_models=AveragingModels(models=( ENet, GBoost, KRR,lasso) )



score=rmsle_cv(averaging_models)

print("\naveraging_models score mean {:.4f} and Std {:.4f}".format(score.mean(),score.std()))
class StackingAveragedModels(BaseEstimator, RegressorMixin):

    def __init__(self, base_models ,meta_model,n_folds=5):

        self.base_models=base_models

        self.meta_model=meta_model

        self.n_folds=n_folds

    

    # we again fit the data on the clone of the original models

    def fit(self,X,y):

        self.base_models_= [list() for x in self.base_models]

        self.meta_model_=clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds ,shuffle=True,random_state=158)

        

        # train cloned base model and create out of fold predictions

        # which are used to train meta_model

        

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        

        for i , model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance=clone(model)

                self.base_models_[i].append(instance)

                instance.fit( X[train_index] ,y[train_index] )

                y_pred= instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index,i]=y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

    

    

    #Do the predictions of all base models on the test data and use the averaged predictions as 

    #meta-features for the final prediction which is done by the meta-model

    def predict(self, X):

        meta_features = np.column_stack([np.column_stack([model.predict(X) for model in base_models]).mean(axis=1) 

                                         for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)
# stacking average model score

stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR), meta_model =lasso)



score = rmsle_cv(stacked_averaged_models)

print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
def rmsle(y,y_pred):

    return np.sqrt(mean_squared_error(y,y_pred))

    
# final training and predictions
stacked_averaged_models.fit(train.values, y_train)

stacked_train_pred = stacked_averaged_models.predict(train.values)

stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))

print(rmsle(y_train, stacked_train_pred))
model_xgb.fit(train, y_train)

xgb_train_pred = model_xgb.predict(train)

xgb_pred = np.expm1(model_xgb.predict(test))

print(rmsle(y_train, xgb_train_pred))
model_lgb.fit(train, y_train)

lgb_train_pred = model_lgb.predict(train)

lgb_pred = np.expm1(model_lgb.predict(test.values))

print(rmsle(y_train, lgb_train_pred))
'''RMSE on the entire Train data when averaging'''



print('RMSLE score on train data:')

print(rmsle(y_train,stacked_train_pred*0.70 +

               xgb_train_pred*0.15 + lgb_train_pred*0.15 ))
ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
ensemble[:50]
result=pd.DataFrame()

result['Id']=test_id

result['SalePrice']=ensemble

result.to_csv('submission12.csv',index=False)
import pickle

with open('my_dumped_model.pkl', 'wb') as fid:

    pickle.dump(model_lgb, fid) 
import pickle

with open('my_dumped_model.pkl', 'rb') as fid:

    loaded_model = pickle.load(fid)