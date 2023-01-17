import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
#gives the shape of the data 
df.shape
#gives the first few values of the data 
df.head()
# shows a heatmap of the data with null values separated out in yellow color
sns.heatmap(df.isnull(),yticklabels=False,cbar=True)
#gives info about the data - # of null records per column & type of data
df.info()
# Lists column names by data type - Integars and Objects
g = df.columns.to_series().groupby(df.dtypes).groups
print(g)
#count the number of null values in each column
df.isnull().sum()
# plots the columns with null values. Number of null values on y-axis and column headers in x-axis
null_counts = df.isnull().sum()
plt.figure(figsize=(16,8))
plt.xticks(np.arange(len(null_counts))+0.5,null_counts.index,rotation='vertical')
plt.ylabel('fraction of rows with missing data')
plt.bar(np.arange(len(null_counts)),null_counts)
# Replaxe null values with Mean of LotFrontage
df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
#df.head()
df.isnull().sum()
#drops some columns - columns with too many null values - number of columns fall by 3 (from 81 to 78)
df.drop(['Alley', 'PoolQC', 'Fence'],axis=1,inplace=True)
df.shape
#replaces all the null values with the mode since the data type is a categorical value
df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])      
df['BsmtFinType1']=df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])  
df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])   
df.shape
# Fill missing values of GarageYrBuilt with corresponding values from YearBuilt
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
# axis=1 when you want to refer to a column. Drop another column with too many null values 
df.drop(['MiscFeature'],axis=1,inplace=True)
df.shape
df['Fireplaces']=df['Fireplaces'].fillna(df['Fireplaces'].mode()[0]) 
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0]) 
null_counts = df.isnull().sum()
plt.figure(figsize=(16,8))
plt.xticks(np.arange(len(null_counts))+0.5,null_counts.index,rotation='vertical')
plt.ylabel('fraction of rows with missing data')
plt.bar(np.arange(len(null_counts)),null_counts)
#gives info about the data - we are looking to see if null values remain
df.info()
#drops the 'Id' column 
df.drop(['Id'],axis=1,inplace=True)
df.shape
df.info()
# Drop all rows with zero values - 1460 rows come down to 1451  
#df = df.dropna(how ='any',axis=0) 
#df.shape
#count and graphs the number of null values
null_counts = df.isnull().sum()
plt.figure(figsize=(16,8))
plt.xticks(np.arange(len(null_counts))+0.5,null_counts.index,rotation='vertical')
plt.ylabel('fraction of rows with missing data')
plt.bar(np.arange(len(null_counts)),null_counts)
sns.heatmap(df.isnull(),yticklabels=False,cbar=True, cmap='coolwarm')
#Handle Categorical Features
#list(df.columns)
#df.info()
#df._get_numeric_data()
#df.select_dtypes(include=['int']).dtypes
col = df.select_dtypes(include=['object'])
#df.select_dtypes(include=['float'])
list(col)
# Assigning column names with categorical features to a variable called "columns'
columns=['MSZoning',
 'Street',
 'LotShape',
 'LandContour',
 'Utilities',
 'LotConfig',
 'LandSlope',
 'Neighborhood',
 'Condition1',
 'Condition2',
 'BldgType',
 'HouseStyle',
 'RoofStyle',
 'RoofMatl',
 'Exterior1st',
 'Exterior2nd',
 'MasVnrType',
 'ExterQual',
 'ExterCond',
 'Foundation',
 'BsmtQual',
 'BsmtCond',
 'BsmtExposure',
 'BsmtFinType1',
 'BsmtFinType2',
 'Heating',
 'HeatingQC',
 'CentralAir',
 'Electrical',
 'KitchenQual',
 'Functional',
 'FireplaceQu',
 'GarageType',
 'GarageFinish',
 'GarageQual',
 'GarageCond',
 'PavedDrive',
 'SaleType',
 'SaleCondition']
#tells me how many colums there are with categorical values 
len(columns)
# Defining a function to convert all categorical values into numerical values. 
# Using the method of One Hot Encoding, create dummy columns for each categorical value using pd.get_dummies function
# Create 

def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:     
        print(fields)
        
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
          
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final
# Creating a copy of df to ensure any changes can be tracked back
main_df=df.copy()
df.shape
main_df.shape
main_df.info()
# Reading the test data file, that has been transformed and saved in kaggle notebook - kernel2dc1902644  - TEST DATA
test_df = pd.read_csv('../input/retestdata0522/reformulated_test.csv')
test_df.shape
#test_df.head()
# Concatinating test and train date across rows - 1434 + 1451 = 2885 rows and 76 columns should be the result
final_df=pd.concat([df,test_df],axis=0)
#tells me the number of rows and columns and the data 
final_df.shape
#final_df.info()
# Checking to see where test and train data merge 
final_df.iloc[1440:1470]
final_df.tail()
# Checking 76 columns to confirm "SalePrice" is included. It should be part of train data but not test data
final_df.columns
# Call the One Hot Multcols function and pass it all categorical columns listed earlier under variable "columns"
final_df=category_onehot_multcols(columns)
# Final data frame now has 237 columns - one column for each categorical value of each column
final_df.info()
# Remove any  duplicate columns 
final_df =final_df.loc[:,~final_df.columns.duplicated()]
# # of columns come down to 177
final_df.info()
final_df
final_df.iloc[1450:1470]
# Dividing the dataframe into training and test data frame
df_Train=final_df.iloc[:1460,:]
df_Test=final_df.iloc[1460:,:]
# Checking to see df_Test has 'SalePrice'with all null values - thereby confirming the right division of final_df into test and train
null_counts = df_Test.isnull().sum()
plt.figure(figsize=(24,8))
plt.xticks(np.arange(len(null_counts))+0.75,null_counts.index,rotation='vertical')
plt.ylabel('rows with missing data')
plt.bar(np.arange(len(null_counts)),null_counts)
# Checking to see df_Train.. clearly no column with 1400+ null values 
null_counts = df_Train.isnull().sum()
plt.figure(figsize=(24,8))
plt.xticks(np.arange(len(null_counts))+0.75,null_counts.index,rotation='vertical')
plt.ylabel('rows with missing data')
plt.bar(np.arange(len(null_counts)),null_counts)
df_Train.head()
df_Train.tail()
df_Test.head()
df_Test.columns
df_Train.columns
len(df_Test.columns)
len(df_Train.columns)
df_Test['SalePrice']
df_Train['SalePrice']
df_Test.tail()
df_Test.shape
df_Train.shape
# Dropping the SalePrice column in df_Test as all values are null
df_Test.drop(['SalePrice'],axis=1,inplace=True)
df_Test.shape
df_Train.shape
# Dividing df_Train into X_train and Y_Train.
# In X_train, we drop SalesPrice and Y_Train has only SalePrice as a column
X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']
# Importing the XGBoost algorithm and using it to fit X_train and y_train
import xgboost
classifier=xgboost.XGBRegressor()
classifier.fit(X_train,y_train)
df_Test.shape
# Using the trained claasifier to predict SalePrice for df_TEst
y_pred=classifier.predict(df_Test)
y_pred
y_pred.shape
# Creating submission file. File will be in Kaggle / Data / Output

pred=pd.DataFrame(y_pred)
sub_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets
#datasets = datasets.iloc[:1459,:]
#datasets['Id'] = datasets['Id'].astype('int32') 
datasets.to_csv('Seventh_sample_submission.csv',index=False)
#datasets.tail(30)
datasets.shape
# Import RandomizedSearchCV to find the best Regressor for the data set.
from sklearn.model_selection import RandomizedSearchCV
regressor=xgboost.XGBRegressor()
# Using the regressor with the parameters derived by RandomCV
regressor=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints=None,
             learning_rate=0.1, max_delta_step=0, max_depth=2,
             min_child_weight=1, missing=None, monotone_constraints=None,
             n_estimators=900, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
             validate_parameters=False, verbosity=None)
regressor.fit(X_train,y_train)
import pickle
filename = 'finalized_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))
df_Test.shape
# Using regressor to predict df_Test 
y_pred=regressor.predict(df_Test)
y_pred
pred=pd.DataFrame(y_pred)
sub_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
#datasets = datasets.iloc[:1459,:]
#datasets['Id'] = datasets['Id'].astype('int32') 
datasets.to_csv('Tuned_Fifth_sample_submission.csv',index=False)
pred
# Assign SalePrice as name of column
pred.columns=['SalePrice']
pred
df_Train.columns
# temp_df copied with values from df_train (SalePrice)
temp_df=df_Train['SalePrice'].copy()
temp_df.head()
# Assign column name "SalePrice"
temp_df.column=['SalePrice']
temp_df.head
temp_df.shape
df_Train.head()
df_Test.head()
df_Train.drop(['SalePrice'],axis=1,inplace=True)
df_Train.shape
# Add temp_df to df_Train - ie. add SalePrice column to df_Train
df_Train=pd.concat([df_Train,temp_df],axis=1)
df_Train['SalePrice']
df_Train.shape
df_Test.shape
# Add predicted values of SalePrice to df_Test
df_Test=pd.concat([df_Test,pred],axis=1)
df_Test.shape
# Create df_Train combining df_Train and df_Test. Df_Train now has 2919 rows for model fitment
df_Train2=pd.concat([df_Train,df_Test],axis=0)
df_Train2.shape
# Create X_train and Y_Train for training model - this time with 2919 rows 
X_train=df_Train2.drop(['SalePrice'],axis=1)
y_train=df_Train2['SalePrice']
X_train.shape
y_train
# Using CVbest Estimator for regressor 
regressor=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints=None,
             learning_rate=0.1, max_delta_step=0, max_depth=2,
             min_child_weight=1, missing=None, monotone_constraints=None,
             n_estimators=900, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
             validate_parameters=False, verbosity=None)
# Using Regressor to fit X_train and Y-Train
regressor.fit(X_train,y_train)
df_Test.drop(['SalePrice'],axis=1,inplace=True)
df_Test.shape
# Using regressor to predict SalePrice values of df_Test
y_pred=regressor.predict(df_Test)
pred=pd.DataFrame(y_pred)
sub_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
#datasets = datasets.iloc[:1459,:]
#datasets['Id'] = datasets['Id'].astype('int32') 
datasets.to_csv('Mashup_sample_submission.csv',index=False)
datasets