# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path='/kaggle/input/house-prices-advanced-regression-techniques/'
train_df=pd.read_csv(path+'train.csv')
test_df=pd.read_csv(path+'test.csv')
sample_submission_df=pd.read_csv(path+'sample_submission.csv')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Construction of a function to plot, and display the total number of the null values
def check_nulls(df):
  sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis') #Plot
  null_columns=df.columns[df.isnull().any()]
  df[null_columns].isnull().sum()
  print(df[null_columns].isnull().sum()) #to display sum of nulls
#Displaying the columns having only NaNs
#train_df.isnull().sum()
null_columns=train_df.columns[train_df.isnull().any()]
#null_columns
train_df[null_columns].isnull().sum()
#For training dataset
check_nulls(train_df)
#Similarly for testing dataset,
check_nulls(test_df)
#construction of function to handle the null values

def null_data_management(train_df):
  null_columns=train_df.columns[train_df.isnull().any()] #all the null columns 
  train_df_nans=train_df[null_columns] #(only NaN containing colums in train_df: these columns contains dtypes= cat/int/float)

  #can be check by: train_df_nans.info()

  train_df_nans_cat_features=train_df_nans.select_dtypes(include=['object']).copy()
  train_df_nans_float_features=train_df_nans.select_dtypes(include=['float64']).copy()

  #train_df_nans_cat_features.columns 
  #train_df_nans_float_features.columns
  #train_df_nans_cat_features.isnull().sum() #shows number of NaNs presenting in each columns
  #the meaning of train_df_nans_cat_features.isnull().sum() is same with,
  # train_df[train_df_nans_cat_features.columns].isnull().sum() 

  for i, t in enumerate(train_df_nans_cat_features):            #to replace categorical
    #print(t)
    if train_df[t].isnull().sum()>=100:                         #if more than 100, dropout
      train_df.drop([t],axis=1,inplace=True) 
    else:
      train_df[t]=train_df[t].fillna(train_df[t].mode()[0])     #replace with mode

  for i, t in enumerate(train_df_nans_float_features):          #to replace float
    #print(t)
    if train_df[t].isnull().sum()>=100:
      train_df.drop([t],axis=1,inplace=True) 
    else:
      train_df[t]=train_df[t].fillna(train_df[t].mean())        #replace with mean()
  return train_df
#Lets replace/remove nulls in training data
train_df=null_data_management(train_df)
check_nulls(train_df)
#Lets replace/remove nulls for test data
test_df=null_data_management(test_df)
check_nulls(test_df)
#Lets remove 'id' columns from both train and testdataset
train_df.drop(['Id'], axis=1, inplace=True)
test_df.drop(['Id'], axis=1, inplace=True)
final_df=pd.concat([train_df, test_df], axis=0)
#Finding the columns that consist the categorical variable
final_cat_columns=final_df.select_dtypes(include=['object']).copy()
multicol=final_cat_columns.columns

def category_onehot_multcols(multicol):
    df_final=final_df
    i=0
    for fields in multicol:
        
        #print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final

df_final=category_onehot_multcols(multicol)
#Many columns are duplicating, therefore, removing the duplicating columns
df_final=df_final.loc[:,~df_final.columns.duplicated()]
df_final.shape
train_df=df_final.iloc[:train_df.shape[0],:]
test_df=df_final.iloc[train_df.shape[0]:,:]

#during concatination, the columns of test dataset may have NaNs, lets drop it.
test_df.drop(['SalePrice'], axis=1,inplace=True)

##Now, lets manage train/test
X_train=train_df.drop(['SalePrice'], axis=1)
y_train=train_df['SalePrice']
import xgboost
regressor=xgboost.XGBRegressor()
booster=['gbtree','gblinear']
base_score=[0.25, 0.5, 0.75,1]

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

from sklearn.model_selection import RandomizedSearchCV
random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)

random_cv.fit(X_train,y_train)
random_cv.best_estimator_
regressor=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=2, min_child_weight=1, missing=None, n_estimators=900,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
regressor.fit(X_train,y_train)
y_pred=regressor.predict(test_df)
#submission preparation
pred=pd.DataFrame(y_pred)
datasets=pd.concat([sample_submission_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('sample1_submission.csv', index=False)
