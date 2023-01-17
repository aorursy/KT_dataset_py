#Import all the necessary Library required for our work

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_squared_error
#Read training & Test csv file in our pandas data frame



data_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv') 

data_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv') 
#Checking the dataset head, as to how the data look like. 

data_train.head()
data_train.dtypes.value_counts()
data_train.describe()
plt.figure(figsize=(30,10));

sns.heatmap(data_train.corr(),annot=True);
#Data is higly right skwed so we will use log transforamtion to remove the skewness of the data. 

#But before that we will try handel the date attribute and do feature enginerring for the date features and lest importatn feature as well. 
plt.figure(figsize=(20,10))

plt.subplot(1, 3, 1)

sns.lineplot(y=data_train['SalePrice'],x=data_train['YearBuilt']- data_train['YrSold'])

plt.subplot(1, 3, 2)

sns.lineplot(y=data_train['SalePrice'],x=data_train['YearBuilt'])

plt.subplot(1, 3, 3)

sns.lineplot(y=data_train['SalePrice'],x=data_train['YrSold'])
#Select features having more than 0.4 correlation value 

data_feature_selection = data_train.corr()['SalePrice'].abs()

data_feature_selection_columns = data_feature_selection[data_feature_selection < 0.1 ].index
data_raw_final = data_train.copy()

data_raw_final['PropertAge']= data_train['YrSold']- data_train['YearBuilt'] # Created PropertyAge feature based on property

                                                                        # Build date and sold date  



#Dropping some more features

data_raw_final.pop('YearBuilt'); #Created new feature Porperty Age so we are dropping YrSold

data_raw_final.pop('YearRemodAdd');#Created new feature Porperty Age so we are dropping YearRemodAdd

data_raw_final.pop('1stFlrSF'); #Strongly correlated to TotalBsmtS

data_raw_final.pop('GarageYrBlt');#Strongly correlated to YearBuilt

data_raw_final.pop('TotRmsAbvGrd');#Strongly correlated to GrLivArea

#Removing least imp Categorical features post multiple run's and analysis



data_raw_final.pop('Exterior1st');#

data_raw_final.pop('Exterior2nd');#

data_raw_final.pop('Condition1');#

data_raw_final.pop('Condition2');#

data_raw_final.pop('GarageType');#

data_raw_final.pop('GarageFinish');#

data_raw_final.pop('PavedDrive');#

data_raw_final.pop('Fence');#

data_raw_final.pop('Utilities');#

data_raw_final.pop('RoofStyle');#

data_raw_final.pop('RoofMatl');#

data_raw_final.pop('Electrical');#

data_raw_final.pop('BsmtFinType1');#

data_raw_final.pop('BsmtFinType2');#













# For loop  to drop the lest correlated feature's in context with sale price

for x in data_feature_selection_columns:

    data_raw_final.drop(x,axis=1,inplace=True)
#Perform the same for Test set



data_test_final = data_test.copy()

data_test_final['PropertAge']= data_test['YrSold']- data_test['YearBuilt'] 



#Dropping some more features

data_test_final.pop('YearBuilt'); #Created new feature Porperty Age so we are dropping YrSold

data_test_final.pop('YearRemodAdd');#Created new feature Porperty Age so we are dropping YearRemodAdd

data_test_final.pop('1stFlrSF'); #Strongly correlated to TotalBsmtS

data_test_final.pop('GarageYrBlt');#Strongly correlated to YearBuilt

data_test_final.pop('TotRmsAbvGrd');#Strongly correlated to GrLivArea      

data_test_final.pop('Exterior1st');#

data_test_final.pop('Exterior2nd');#

data_test_final.pop('Condition1');#

data_test_final.pop('Condition2');#

data_test_final.pop('GarageType');#

data_test_final.pop('GarageFinish');#

data_test_final.pop('PavedDrive');#

data_test_final.pop('Fence');#

data_test_final.pop('Utilities');#

data_test_final.pop('RoofStyle');#

data_test_final.pop('RoofMatl');#

data_test_final.pop('Electrical');#

data_test_final.pop('BsmtFinType1');#

data_test_final.pop('BsmtFinType2');#



# For loop  to drop the lest correlated feature's in context with sale price

for x in data_feature_selection_columns:

    data_test_final.drop(x,axis=1,inplace=True)
#Checking the presence of null value and their respective count in each feature

null_column_name = data_raw_final.columns[data_raw_final.isnull().any()]

data_raw_final[null_column_name].isnull().sum()
data_raw_final[null_column_name].dtypes
#Extract all Object dataypes from the traning dataset

index = data_raw_final.select_dtypes(include=['object'])

for x in index.columns:

    data_raw_final[x].replace(np.nan,value= 'Missing',inplace=True)

    data_raw_final[x] = data_raw_final[x].str.upper()

    print (data_raw_final[x].value_counts(dropna=False));

    

#Dothe same for test set as well, withour printing the result. 

index = data_test_final.select_dtypes(include=['object'])

for x in index.columns:

    data_test_final[x].replace(np.nan,value= 'Missing',inplace=True)

    data_test_final[x] = data_test_final[x].str.upper()

    
grade={'MISSING': 0,'PO': 1,'FA':2,'TA':3 ,'GD':4,'EX':5}                             

data_raw_final['ExterQual']=data_raw_final['ExterQual'].replace(grade)

data_raw_final['ExterCond']=data_raw_final['ExterCond'].replace(grade) 

data_raw_final['HeatingQC']=data_raw_final['HeatingQC'].replace(grade)

data_raw_final['KitchenQual']=data_raw_final['KitchenQual'].replace(grade)

data_raw_final['FireplaceQu']=data_raw_final['FireplaceQu'].replace(grade) 

data_raw_final['PoolQC']=data_raw_final['PoolQC'].replace(grade)   

data_raw_final['BsmtQual']=data_raw_final['BsmtQual'].replace(grade) 

data_raw_final['BsmtCond']=data_raw_final['BsmtCond'].replace(grade) 

data_raw_final['GarageQual']=data_raw_final['GarageQual'].replace(grade)

data_raw_final['GarageCond']=data_raw_final['GarageCond'].replace(grade)
grade={'MISSING': 0,'PO': 1,'FA':2,'TA':3 ,'GD':4,'EX':5}                             

data_test_final['ExterQual']=data_test_final['ExterQual'].replace(grade)

data_test_final['ExterCond']=data_test_final['ExterCond'].replace(grade) 

data_test_final['HeatingQC']=data_test_final['HeatingQC'].replace(grade)

data_test_final['KitchenQual']=data_test_final['KitchenQual'].replace(grade)

data_test_final['FireplaceQu']=data_test_final['FireplaceQu'].replace(grade) 

data_test_final['PoolQC']=data_test_final['PoolQC'].replace(grade)   

data_test_final['BsmtQual']=data_test_final['BsmtQual'].replace(grade) 

data_test_final['BsmtCond']=data_test_final['BsmtCond'].replace(grade) 

data_test_final['GarageQual']=data_test_final['GarageQual'].replace(grade)

data_test_final['GarageCond']=data_test_final['GarageCond'].replace(grade)
data_raw_final.replace(to_replace=np.nan,value=0,inplace=True);

data_test_final.replace(to_replace=np.nan,value=0,inplace=True);
null_column_name = data_test_final.columns[data_test_final.isnull().any()]

data_test_final[null_column_name].isnull().sum()
null_column_name = data_raw_final.columns[data_raw_final.isnull().any()]

data_raw_final[null_column_name].isnull().sum()
data_raw_final = pd.get_dummies(data_raw_final)

data_test_final = pd.get_dummies(data_test_final)
#Find all the missing categorical feature not available in test set and assign zero to it. 

#Using this trick as some of the high cardinality features are bound to have some value which different in Training and Test set



diff = data_raw_final.columns.difference(data_test_final.columns)

print('Index difference')

print(diff)



#Running a loop to add all missing featuresin the Test set

for x in diff:

    if x == 'SalePrice':

        print('-----------------------')

        print('SalePrice Index Ignored')

    else:

        data_test_final[x]=0
#Make copy of the final dataset 

data_scaled = data_raw_final.copy()
# Split the training dataset into two parts training and validation set 

from sklearn.model_selection import train_test_split

y = data_scaled.pop('SalePrice')

x = data_scaled

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1)

from sklearn.linear_model import LinearRegression

regression_model = LinearRegression();

regression_model.fit(X_train, y_train);
print("Linear Model Training  Score   :{0}".format(regression_model.score(X_train,y_train)))

print("Linear Model Validation  Score :{0}".format(regression_model.score(X_test,y_test)))
Feature_imp_df = pd.DataFrame(index=x.columns)

Feature_imp_df['Feature_imp'] = pd.DataFrame(regression_model.coef_,index=x.columns)

Feature_imp_df.sort_values(by='Feature_imp',inplace=True)
plt.figure(figsize=(10,40));

plt.tight_layout();

sns.barplot(y=Feature_imp_df.index,x=Feature_imp_df['Feature_imp']);
from catboost import CatBoostRegressor

catboost_model = CatBoostRegressor()

catboost_model.fit(X_train, y_train)
print("Catboost Model Training  Score   :{0}".format(catboost_model.score(X_train,y_train)))

print("Catboost Model Validation  Score :{0}".format(catboost_model.score(X_test,y_test)))
Feature_imp_Cat_df = pd.DataFrame(index=catboost_model.feature_names_)

Feature_imp_Cat_df['Feature_imp']= pd.DataFrame(catboost_model.feature_importances_,index=catboost_model.feature_names_)

Feature_imp_Cat_df.sort_values(by='Feature_imp',inplace=True)
plt.figure(figsize=(10,45))

plt.tight_layout()

sns.barplot(Feature_imp_Cat_df['Feature_imp'],Feature_imp_Cat_df.index)
#Hyper Parameter Tuning for Catboost to reduce overfitt and generalize the model
model = CatBoostRegressor(grow_policy='Lossguide',per_float_feature_quantization =['0:border_count=1024', '8:border_count=1024'])



grid = {'depth': [2,3,4,5],

        'l2_leaf_reg': [1, 3, 5],

        'random_strength':[2,3],

        'max_leaves':[5,10,15,20]

       }



grid_search_result = model.grid_search(grid, 

                                       X=X_train, 

                                       y=y_train,

                                       plot=True)

print("Catboost Model Training  Score   :{0}".format(model.score(X_train,y_train)))

print("Catboost Model Validation  Score :{0}".format(model.score(X_test,y_test)))
#Preparing the final submission file :

Submission = pd.DataFrame()

Submission['Id'] = data_test['Id']

Submission['SalePrice'] = data_test['SalePrice'] = model.predict(data_test_final)
#Export file to CSV

Submission.to_csv('submission_9.csv',index=False)