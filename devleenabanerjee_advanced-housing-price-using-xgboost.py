#Importing the libraries for Data Manipulation and Visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
%matplotlib inline
matplotlib.style.use('fivethirtyeight')

#Importing the libraries for data modelling and error metrics
from sklearn import preprocessing
lbl = preprocessing.LabelEncoder()
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
#Importing the train and test dataset
train=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

traindata=train.copy()
testdata=test.copy()
train.head()
train.shape
train.info()
features_withna=[features for features in train.columns if train[features].isna().sum()>1]
for features in features_withna:
    print(features,' has ' , train[features].isna().sum(), "  Missing Values out of ", train.shape[0], "\n")
#Removing the features with large number of missing values
train.drop(['MiscFeature','Fence','PoolQC','Alley'], axis=1, inplace=True)
features_withna.remove('MiscFeature')
features_withna.remove('Fence')
features_withna.remove('PoolQC')
features_withna.remove('Alley')
train.columns.str.contains('Yr') |train.columns.str.contains('Year')
temporal_features=[features for features in train.columns if 'Yr' in features or 'Year' in features]
numeric_features=[features for features in train.columns if train[features].dtypes in ['int64', 'float64'] and features not in temporal_features ]
object_features=[features for features in train.columns if train[features].dtypes=='object']

numeric_features
object_features
temporal_features

for features in object_features:
    print(features ,' has ', train[features].nunique(), ' unique values\n')
for features in features_withna:
    if features in object_features:
        train[features].fillna(train[features].mode()[0], inplace=True)
    else:
        train[features].fillna(train[features].median(), inplace=True)        
for features in features_withna:
    print(features, train[features].isna().sum(), "  Missing Values out of ", train.shape[0], "\n")
test.head()
test.shape
train.info()
features_withnatest=[features for features in test.columns if test[features].isna().sum()>1]
for features in features_withnatest:
    print(features, ' has ' , test[features].isna().sum(), "  Missing Values out of ", test.shape[0], "\n")
#Removing the features with large number of missing values
test.drop(['MiscFeature','Fence','PoolQC','Alley'], axis=1, inplace=True)
features_withnatest
features_withnatest.remove('MiscFeature')
features_withnatest.remove('Fence')
features_withnatest.remove('PoolQC')
features_withnatest.remove('Alley')
temporal_features_test=[features for features in test.columns if 'Yr' in features or 'Year' in features]
numeric_features_test=[features for features in test.columns if test[features].dtypes in ['int64', 'float64'] and features not in temporal_features ]
object_features_test=[features for features in test.columns if test[features].dtypes=='object']
for features in object_features_test:
    print(features ,' has ', test[features].nunique(), 'unique values\n')
for features in features_withnatest:
    if features in object_features_test:
        test[features].fillna(test[features].mode()[0], inplace=True)
    else:
        test[features].fillna(test[features].median(), inplace=True)    
for features in features_withnatest:
    print(features, ' has ' , test[features].isna().sum(), "  Missing Values out of ", test.shape[0], "\n")
test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mode()[0], inplace=True)
test['BsmtFullBath'].fillna(test['BsmtFullBath'].mode()[0], inplace=True)
test['Functional'].fillna(test['Functional'].mode()[0], inplace=True)
test['MSZoning'].fillna(test['MSZoning'].mode()[0], inplace=True)
test['Utilities'].fillna(test['Utilities'].mode()[0], inplace=True)
train.head()
## Lets Find the relationship between Categorical Features and Sale PRice

for feature in object_features:
    sns.barplot(x=train[feature],y=train['SalePrice'])
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
## Lets Find the realtionship between Categorical Features and Sale Price

for feature in numeric_features:
    plt.scatter(x=train[feature],y=train['SalePrice'])
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
## Lets Find the realtionship between Temporal Features and Sale PRice

for feature in temporal_features:
    sns.lineplot(x=train[feature],y=train['SalePrice'])
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
## Lets Find the realtionship between Categorical Features and Sale Price

for feature in numeric_features:
    sns.boxplot(y=train[feature])
    plt.ylabel(feature)
    plt.title(feature)
    plt.show()
for feature in numeric_features:
    if 0 in train[feature].unique():
        pass
    else:
        train[feature]=np.log(train[feature])
        train.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()
temporal_features
train['YearBuilt']=train['YrSold']-train['YearBuilt']
train['YearRemodAdd']=train['YrSold']-train['YearRemodAdd']
train['GarageYrBlt']=train['YrSold']-train['GarageYrBlt']
## Here we will compare the difference between All years feature with SalePrice

for feature in temporal_features:
    if feature!='YrSold':
        sns.lineplot(train[feature],train['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()
object_features
train.head()
for feature in numeric_features_test:
    if 0 in train[feature].unique():
        pass
    else:
        test[feature]=np.log(test[feature])
        test.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()
temporal_features_test
test['YearBuilt']=test['YrSold']-test['YearBuilt']
test['YearRemodAdd']=test['YrSold']-test['YearRemodAdd']
test['GarageYrBlt']=test['YrSold']-test['GarageYrBlt']
final_df=pd.concat([train,test],axis=0)

final_df.info()
final_df.head()
object_features=[features for features in final_df.columns if final_df[features].dtypes=='object']

def one_hot_encoding(obj_features):
    final_dfcopy=final_df.copy()
    for features in obj_features:
        print(features)
        df=pd.get_dummies(final_dfcopy[features],drop_first=True)
        df.head()
        final_dfcopy=pd.concat([df,final_dfcopy],axis=1)
    final_dfcopy.drop(obj_features,axis=1,inplace=True)   
    return final_dfcopy
final_df=one_hot_encoding(object_features)
final_df =final_df.loc[:,~final_df.columns.duplicated()]
final_df.shape
final_df.head()
final_df['SalePrice']
final_df.info()
df_Train=final_df.iloc[:1422,:]
df_Test=final_df.iloc[1422:,:]
df_Train.head()
df_Train['SalePrice']
df_Test.head()
X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']
regressor=XGBRegressor()
booster=['gbtree','gblinear']
base_score=[0.25,0.5,0.75,1]
## Hyper Parameter Optimization


n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }
# Set up the random search with 4-fold cross validation
random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)
random_cv.fit(X_train,y_train)

random_cv.best_estimator_
model = XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.05, max_delta_step=0,
             max_depth=2, min_child_weight=4, missing=None, n_estimators=900,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)

regressor.fit(X_train,y_train)
df_Test.drop(['SalePrice'],axis=1,inplace=True)
df_Test.shape

df_Test.head()
df_Test.head()
y_pred=regressor.predict(df_Test)

y_pred
#Create Sample Submission file and Submit using ANN
pred=pd.DataFrame(y_pred)
sub_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
datasets=pd.concat([df_Test['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('AdvancedHousingusingXGBoost.csv',index=False)
pred
