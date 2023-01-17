# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#importing necessary libraries
import csv
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV,cross_val_score
from scipy.stats import norm
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,AdaBoostRegressor)
import matplotlib.pyplot as plt
import scipy
from scipy.special import boxcox1p
from sklearn.metrics import  make_scorer
from sklearn.feature_selection import RFE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

train = pd.read_csv('../input/advance-house-price-predicitons/train.csv')
test = pd.read_csv('../input/advance-house-price-predicitons/test.csv')


#joining train and test data
df=pd.concat((train, test)).reset_index(drop=True)
print(df.head())
#sns.heatmap(df)


#print((df.dtypes[df.dtypes!='object'].index).shape)
#print((df.dtypes[df.dtypes=='object'].index).shape)

#from scipy.stats import norm
#sns.distplot(train['SalePrice'].values,fit=norm);

#from scipy import stats
#import numpy as np
#z = np.abs(stats.zscore(df))
#print(z)
#print(np.where(z > 3))

#print(df.shape)
#filtered_entry = (z < 3).all(axis=1)
##df1=df[filtered_entry]
#print(df1.head())
initial_ytrain_series= train['SalePrice']
print("---------saleprice----------")
sns.distplot(train['SalePrice'] , fit=norm);
plt.show()

initial_ytrain= train['SalePrice']
#df["SalePrice"] = np.log1p(train["SalePrice"])
print("---------saleprice with log----------")


#transforming the traget variable to logarithm scale
ytrain=  np.log1p(train["SalePrice"]);
sns.distplot(ytrain , fit=norm);
plt.show()


#removing ID column from test and train
train_id=train["Id"]
test_id=test["Id"]

#removing sale price and ID from dataset 
df.drop('SalePrice',axis=1,inplace=True)
df.drop('Id',axis=1,inplace=True)


df = df.reset_index()

#print(train.shape,test.shape)
#print(test.describe())
#print(train.head())

#print(train.describe())
#print(train.columns)
#print(test.dtypes)


# function to determine the null value percent
def nullvalues(df):
    total_null_values = df.isnull().sum()
    total_values = len(df)
    null_values_percent = ((total_null_values*100)/total_values).sort_values(ascending=False)[:15]
    #table = pd.concat((total_null_values, null_values_percent), axis=1)
    #print(table[table[1]>0])
    #print(table.sort_values(by=1,ascending= False))
    missing_data = pd.DataFrame({'Missing Value Percent' :null_values_percent})
    f, ax = plt.subplots(figsize=(15,10))
    plt.xticks(rotation='65')
    sns.barplot(null_values_percent.index, y=(null_values_percent))
    sns.set(style="whitegrid")
    sns.set_color_codes("pastel")
    plt.xlabel('Features')
    plt.ylabel('Percent - missing values')
    plt.title('Percent missing data')
    plt.show()
    print(missing_data)
    

print(nullvalues(df))


#combine few variables into string type
df['MSSubClass'] = df['MSSubClass'].astype(str)
df['OverallQual'] = df['OverallQual'].astype(str)
df['OverallCond'] = df['OverallCond'].astype(str)
df['YrSold'] = df['YrSold'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)


#filling null column with "None" 

none_columns=['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageType', 'GarageFinish',
              'GarageQual', 'GarageCond','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
              'MasVnrType','MSSubClass']

for c in none_columns:
    df[c]=df[c].fillna("None")
    
    
#filling missing value with 0 value
    
zero_columns=['GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 
              'BsmtFullBath', 'BsmtHalfBath','MasVnrArea'] 
for z in zero_columns:
    train[z]=train[z].fillna(0)
    #df.drop(z,inplace=True,axis=1)
    

#filling null column with "mode" 

mode_columns= ['MSZoning','Electrical','KitchenQual','Exterior1st','Exterior2nd','SaleType','Functional'
              ,'LotFrontage']

for m in mode_columns:
    df[m]=df[m].fillna(train[m].mode()[0])
    

print(df.shape)    

#Label Encoding categorical variable

categorical_variable=['GarageFinish','BsmtCond','BsmtQual',  'GarageQual', 'GarageCond','BldgType', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 'HouseStyle',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope','RoofStyle',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
       'YrSold', 'MoSold','MSZoning','LandContour','LotConfig','Neighborhood','Condition1','Condition2',
                     'RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation','Heating','HeatingQC',
                     'CentralAir','Electrical','KitchenQual','Functional','GarageType','FireplaceQu',
                     'GarageCond','PavedDrive','MiscFeature','SaleType','SaleCondition','OverallQual']

for c in  categorical_variable:
    df[c]= LabelEncoder().fit_transform(df[c].values)
    

#print(df.head())
    

#dropping the utility column    
df.drop('Utilities',axis=1,inplace=True)


#finding the skewness of the predictors in the dataset

numeric_features=df.dtypes[df.dtypes!='object'].index
#print(numeric_features)
skewValue = df[numeric_features].skew().sort_values(ascending=False)
skewness=pd.DataFrame({'Skew':skewValue})
positive_skew=skewness[skewness['Skew']>1]
print("-------------")
print(positive_skew)
postive_skew_index=positive_skew.index

#sns.barplot(postive_skew_index, y=skewValue )
#print(postive_skew_index)
#print(positive_skew)
#print(skewValue.sort_values(ascending=False))


#applying boxcox transformation to remove skewness
lam=0.15

for s in postive_skew_index:
    df[s]=boxcox1p(df[s],lam)
    #print(s,df[s].skew())

print("--------------")

#finding skewness after applying boxcox transformation


skewValue2 = df[numeric_features].skew().sort_values(ascending=False)
#psotive_skew= skewValue2[skewValue2]
skewness2=pd.DataFrame({'Skew':skewValue2})
skewness2_index=skewness2.index
print(skewness2)


#skewValue2 = df[numeric_features].skew().sort_values(ascending=False)
#skewness2=pd.DataFrame({'Skew':skewValue})
#print(skewness2)


#Parameter Grid- which would have been otherwise used with GRIDSearchCV or RandomisedSearch CV
'''
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 500, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
#max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

rfrandom_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf
               }

learning_rate = [0.1,0.5,1.0]
#print(random_grid)

adarandom_grid = {'n_estimators': n_estimators,
                  'learning_rate': learning_rate
}

xgb_randomgrid = {
    'learning_rate' :learning_rate,
    'n_estimators':n_estimators,
    'max_depth':max_depth
}
'''

#np.where(train.values >= np.finfo(np.float64).max)

rf=RandomForestRegressor()
ada=AdaBoostRegressor()
gb=GradientBoostingRegressor()
xgb=XGBRegressor()

# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
#cv_split = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
cv_split = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

#xgb=XGBRegressor(n_estimators=50,learning_rate=0.05)   
#xgb_random = RandomizedSearchCV(estimator=xgb,param_distributions=xgb_randomgrid,cv=cv_split)
#rf_random = RandomizedSearchCV(estimator=rf,param_distributions=rfrandom_grid,cv=cv_split)
#gb_random=RandomizedSearchCV(estimator=gb,param_distributions=rfrandom_grid,cv=cv_split)
#ada_random = RandomizedSearchCV(estimator=ada,param_distributions=adarandom_grid,cv=cv_split)

from sklearn import linear_model
poly=PolynomialFeatures(order=2)
lm= LinearRegression()
rf=RandomForestRegressor()
ls = linear_model.Lasso(alpha=0.1)


#rf=RandomForestRegressor(max_features='auto',min_samples_leaf=2,n_estimators=50)
#ada=AdaBoostRegressor(n_estimators=50,learning_rate=0.05)
#gb=GradientBoostingRegressor(n_estimators=250, learning_rate=0.1, max_depth=5, random_state=0, loss='ls')
#xgb=XGBRegressor(n_estimators=50,learning_rate=0.05)   



print(np.where(df.values >= np.finfo(np.float64).max))
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df[:] = np.nan_to_num(df)


#caluclating corrleation cofficient by first combining the dataset again
corelation_df = pd.concat((train,ytrain), axis=1)

#remvong the duplicate column of SalePrice
corelation_df = corelation_df.loc[:,~corelation_df.columns.duplicated()]

#plotting heatmap of all variables- sale price varibale is dropped in it
sns.heatmap(df)

train = df[:train.shape[0]]
test = df[train.shape[0]:]

#print(train.shape,ytrain.shape)

#print(corelation_df.head())
df.reset_index(inplace=True)


corr=corelation_df.corr()
#corr_data = pd.DataFrame({'Corr coff' :corr})

#sns.heatmap(corr,cmap="YlGnBu", annot=True)

sns.heatmap(corr,cmap="YlGnBu")

#locking the sales price column for all rows
new_corr_df=pd.DataFrame(corr.loc[:,"SalePrice"])
#print(new_corr_df)


#plotting correlation coff with just sales price variable
f, ax = plt.subplots(figsize=(15,10))
plt.xticks(rotation='65')
sns.heatmap(new_corr_df)
plt.show()


#putting the columns with correlation cofficient of less than 0.2 and 0.3 in two lists

non_corr_columns_02= ["MSSubClass","OverallCond","BsmtFinSF2","LowQualFinSF","BsmtHalfBath","BedroomAbvGr","KitchenAbvGr","EnclosedPorch","3SsnPorch","ScreenPorch",
                   "PoolArea","MiscVal","MoSold","YrSold"]

non_corr_columns_03= ["MSSubClass","OverallCond","BsmtFinSF2","LowQualFinSF","BsmtHalfBath","BedroomAbvGr","KitchenAbvGr","EnclosedPorch","3SsnPorch","ScreenPorch",
                   "PoolArea","MiscVal","MoSold","YrSold","WoodDeckSF","2ndFlrSF","OpenPorchSF","HalfBath","LotArea","GarageYrBlt","BsmtFullBath","BsmtUnfSF"]


#removing the columns with corellation coff less than 0.2

for nc in non_corr_columns_02:
    train.drop(nc,inplace=True,axis=1)
    test.drop(nc,inplace=True,axis=1)

    
    
#print("Program has come here")
numeric_features=train.dtypes[df.dtypes!='object'].index
#print(numeric_features)

#print(train.shape,"-----------------------------")

#performing RFE on different models
rfe_selector1 = RFE(estimator=ls, step=10, verbose=5,n_features_to_select=65)
rfe_selector_rf = RFE(estimator=rf, step=10, verbose=5,n_features_to_select=65)
rfe_selector_gb = RFE(estimator=gb, step=10, verbose=5,n_features_to_select=65)
rfe_selector_xgb = RFE(estimator=xgb, step=10, verbose=5,n_features_to_select=65)

ridge= Ridge(alpha=0.1)


#poly=PolynomialFeatures(2)
#train=poly.fit_transform(train)

#removing outliers from the dataset by replacing their values

for i in train.columns:
    quartile_one,quartile_three = np.percentile(train[i],[25,75])
    quartile_first,quartile_last = np.percentile(train[i],[12,85])
    inter_quartile_range = quartile_three-quartile_one
    cut_off= 1.5*inter_quartile_range
    lower_bound = quartile_one - (cut_off)
    upper_bound = quartile_three + (cut_off)
    print()
    train[i].loc[df[i] < lower_bound] = quartile_first
    train[i].loc[df[i] > upper_bound] = quartile_last
    print(i,lower_bound,upper_bound,quartile_first,quartile_last )

        

#building voting regressor of different kind 

#eclf1 = VotingRegressor(estimators=[('rm',rf_random), ('ada', ada_random),('XGB',xgb_random),('GB',gb_random)])

eclf2 = VotingRegressor(estimators=[('rm',rf),('XGB',xgb),('ada', ada),('GB',gb)])

eclf4 = VotingRegressor(estimators=[('rm_rfe',rfe_selector_rf),('XGB_rfe',rfe_selector_xgb),('GB_rfe',rfe_selector_gb)])

eclf3 = VotingRegressor(estimators=[('rm',rf), ('XGB',xgb),('GB',gb)])

eclf4= VotingRegressor(estimators=[('rm',rf),('XGB',xgb),('ada', ada),('GB',gb),('rm_rfe',rfe_selector_rf),('XGB_rfe',rfe_selector_xgb),('GB_rfe',rfe_selector_gb),
                      ("Linear Regression",lm),('Lasso',ls)])

eclf5 = VotingRegressor(estimators=[('rm',rf), ('Linear Regression',lm),('XGB',xgb), ('GB',gb)])


#to print the test and training sample splitted by the cv
#for train_index, test_index in cv_split.split(train):
#     print("TRAIN:", train_index, "TEST:", test_index)


#print(train.shape,ytrain.shape)


#using voting regressor and other models to obtain the accuracy and standar deviation

for clf, label in zip([lm,rf,ada,ls,gb,ridge,xgb,eclf2,eclf4, eclf3,eclf5], ['Linear Regression','RandomForest','ADAboost', 'Lasso','GB','Ridge','XGB','Combine with ada',
                                                                      'Combine without ada with rfe','Combine without ada',"combine with linear"]):
                     score=cross_val_score(clf,train,ytrain,scoring='neg_mean_squared_error',cv=cv_split) 
                     clf.fit(train,ytrain)
                     #train['prediction %s'%label]=clf.predict(train) # this line was a mistake which resulted in additon of a column in dataset and increased the accuracy of the model, but was later corrected
                     print("Accuracy: %0.3f (+/- %0.3f) [%s]" % (score.mean(), score.std(), label))  
                     print('----------------')
    
    
#predicting values of traget variable in test
    
#yhat3=np.expm1(eclf3.predict(test)) #without ada
#yhat2=np.expm1(eclf2.predict(test)) #with ada
#yhat4=np.expm1(eclf4.predict(test)) #with everything
yhat5=np.expm1(eclf5.predict(test)) #with linear
#yhat2=(eclf.predict(train))
#yhat=pd.Series(yhat)
#print(yhat)

submission = pd.DataFrame()
submission['Id'] = test_id
submission['SalePrice'] = yhat5
submission.to_csv('submission.csv',index=False)

#kaggle competitions submit -c house-prices-advanced-regression-techniques -f submission.csv -m "submit"

