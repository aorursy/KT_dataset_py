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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
df_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
df_train.head()
train_Id = df_train["Id"]
test_Id = df_test["Id"]
df_train.describe()
df_test.head()
df_test.describe()
sns.heatmap(df_train.isnull())
sns.heatmap(df_test.isnull())
mat=df_train.corr()
fig,ax = plt.subplots(figsize = (30,30))
sns.heatmap(mat,annot = True, annot_kws={'size': 12})
#After analyzing heatmap we can conclude that saleprice has maximum dependency on these 4 features
abc=df_train[["GarageCars","OverallQual","GrLivArea","GarageArea","SalePrice"]]
sns.pairplot(abc)
df_train.info()
#we can see object(43),which means 43 categorical data columns and rest are numeric
df_test.info()
df_train.isnull().sum().sort_values(ascending=False)[0:20]
df_test.isnull().sum().sort_values(ascending=False)[0:35]
total_cells = np.product(df_train.shape)
total_missing = df_train.isnull().sum().sum()
print('percentage of data that is missing = ',(total_missing/total_cells)*100)
#deleting those columns which have more than 50% Nan values
#as those columns are same for both test and train datas
list_drop=["PoolQC","MiscFeature","Alley","Fence","GarageYrBlt"]

for col in list_drop:
    del df_train[col]
    del df_test[col]
df_train.isnull().sum().sort_values(ascending=False)[0:15]
df_test.isnull().sum().sort_values(ascending=False)[0:30]
df_train.LotFrontage.value_counts(dropna=False)
df_train.LotFrontage.fillna(df_train.LotFrontage.mean(),inplace=True)
df_test.LotFrontage.fillna(df_test.LotFrontage.mean(),inplace=True)
df_train.shape
list_fill_train=["BsmtCond", "BsmtQual", "GarageType", "GarageCond", "GarageFinish",
                 "GarageQual","MasVnrType","BsmtFinType2","BsmtExposure","FireplaceQu","MasVnrArea"]

for j in list_fill_train:
    #df_train[j].fillna(df_train[j].mode(),inplace=True)
    # wrong way to do it.
    df_train[j] = df_train[j].fillna(df_train[j].mode()[0])
    df_test[j] = df_test[j].fillna(df_train[j].mode()[0])   #.mode() returns tuple of mode and frequency ,so we use [0] to get mode only
    
df_train.shape
print(df_train.isnull().sum().sort_values(ascending=False)[0:5])
print(df_test.isnull().sum().sort_values(ascending=False)[0:20])
df_train.dropna(inplace=True)
#Dropping 37 rows of BsmtFinType1 and 1 of Electrical 
df_train.shape
list_test_str = ['BsmtFinType1', 'Utilities','BsmtFullBath', 'BsmtHalfBath', 'Functional', 'SaleType', 'Exterior2nd', 
           'Exterior1st', 'KitchenQual','MSZoning']
list_test_num= ['GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',  'TotalBsmtSF', 'GarageArea',]

for item in list_test_str:
    df_test[item] = df_test[item].fillna(df_test[item].mode()[0])
for item in list_test_num:
    df_test[item] = df_test[item].fillna(df_test[item].mean())
print(df_train.isnull().sum().sort_values(ascending=False)[0:5])
print(df_test.isnull().sum().sort_values(ascending=False)[0:5])
df_test.shape
del df_train["Id"]
del df_test["Id"]
print(df_train.shape)
print(df_test.shape)
print(df_train.isnull().any().any())
print(df_test.isnull().any().any())
#.any() returns true or false for each column in dataframe
#.any().any() returns true or false for entire dataframe
#joining data sets
df_final=pd.concat([df_train,df_test],axis=0)
df_final.info()
df_final.shape
columns = ['MSZoning', 'Street','LotShape', 'LandContour', 'Utilities',
           'LotConfig', 'LandSlope','Neighborhood', 'Condition1', 'Condition2',
           'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
           'Exterior2nd', 'MasVnrType','ExterQual', 'ExterCond', 'Foundation',
           'BsmtQual', 'BsmtCond','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
           'Heating', 'HeatingQC', 'CentralAir', 'Electrical','KitchenQual',
           'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
           'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']

#39 categorical column which has to be converted into numeric data 
def One_hot_encoding(columns):
    final_df=df_final
    i=0
    for fields in columns:
        df1=pd.get_dummies(df_final[fields],drop_first=True)
        
        df_final.drop([fields],axis=1,inplace=True)         
        if i==0:                                            
            final_df=df1.copy()
        else:           
            final_df=pd.concat([final_df,df1],axis=1)  
        i=i+1
       
        
    final_df=pd.concat([df_final,final_df],axis=1) 
        
    return final_df
main_df=df_train.copy()
df_final.head()
df_final = One_hot_encoding(columns)
df_final.head()
df_final.shape 
df_final =df_final.loc[:,~df_final.columns.duplicated()]
df_final.shape
df_train_m=df_final.iloc[:1422,:]
df_test_m=df_final.iloc[1422:,:]
df_test_m.shape
df_test_m.drop(["SalePrice"],axis=1,inplace=True)
df_test_m.shape
x_train_final=df_train_m.drop(["SalePrice"],axis=1)
y_train_final=df_train_m["SalePrice"]
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
X_train, X_test, Y_train, Y_test = train_test_split(x_train_final, y_train_final, random_state=1)
print(X_train.shape,X_test.shape)
print(Y_train.shape,Y_test.shape)
#model building
linear_reg=LinearRegression()
linear_reg.fit(X_train,Y_train)
Y_pred = linear_reg.predict(X_test)
print("R-Squared Value for Training Set: {:.3f}".format(linear_reg.score(X_train,Y_train)))
print("R-Squared Value for Test Set: {:.3f}".format(linear_reg.score(X_test,Y_test)))
print(r2_score(Y_test, Y_pred))
y_pred_test=linear_reg.predict(df_test_m)
pred_df = pd.DataFrame(y_pred_test, columns=['SalePrice'])
test_id_df = pd.DataFrame(test_Id, columns=['Id'])
from sklearn.ensemble import RandomForestRegressor
R_forest=RandomForestRegressor()
R_forest.fit(X_train,Y_train)
print("R-Squared Value for Training Set: {:.3f}".format(R_forest.score(X_train,Y_train)))
print("R-Squared Value for Test Set: {:.3f}".format(R_forest.score(X_test,Y_test)))
y_pred_rforest_test=R_forest.predict(df_test_m)
pred_rforest_df = pd.DataFrame(y_pred_rforest_test, columns=['SalePrice'])
from xgboost import XGBRegressor
xgb=XGBRegressor()
xgb.fit(X_train,Y_train)
y_pred_xgb_test=xgb.predict(df_test_m)
pred_xgb_df = pd.DataFrame(y_pred_xgb_test, columns=['SalePrice'])
submission1 = pd.concat([test_id_df, pred_xgb_df], axis=1)
submission1.head()
submission1.to_csv(r'submission.csv', index=False)
#parameters
n_estimators = [100,300,500,700,900,1100,1300,1500]
max_depth = [2,3,5,10,15,20,25]
learning_rate = [0.05,0.1,0.15,0.20]
booster = ['gbtree','gblinear']
min_child_weight = [1,2,3,4]
base_score = [0.25,0.5,0.75,1.0]
hyperparameter_grid={
    'n_estimators' : n_estimators,
    'max_depth' :  max_depth,
    'learning_rate' : learning_rate,
    'booster' : booster,
    'min_child_weight' : min_child_weight,
    'base_score' : base_score
} 
from sklearn.model_selection import RandomizedSearchCV
random_cv=RandomizedSearchCV(estimator=xgb,
                            param_distributions=hyperparameter_grid,
                            cv=5,
                            n_jobs=3,
                            random_state=5,
                             n_iter=50,
                            scoring='neg_mean_absolute_error')
random_cv.fit(X_train,Y_train)
random_cv.best_estimator_
xgb=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=3,
             min_child_weight=1,monotone_constraints='()',
             n_estimators=1100, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
xgb.fit(X_train,Y_train)
y_pred_xgb_test=xgb.predict(df_test_m)
pred_xgb_df = pd.DataFrame(y_pred_xgb_test, columns=['SalePrice'])
submission1 = pd.concat([test_id_df, pred_xgb_df], axis=1)
submission1.head()
submission1.to_csv(r'submission1.csv', index=False)