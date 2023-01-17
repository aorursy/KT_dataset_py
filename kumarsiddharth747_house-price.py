# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import re

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df  = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test  = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

#print(df.shape)

#print(df.columns)

sns.heatmap(df.isnull(),yticklabels=False,cbar=False)

df.shape
#missing data_train

total = df.isnull().sum().sort_values(ascending=False)

percent = ((df.isnull().sum()/1460)*100).sort_values(ascending=True)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)

#missing data_test

total_test = df_test.isnull().sum().sort_values(ascending=False)

percent_test = ((df_test.isnull().sum()/1460)*100).sort_values(ascending=True)

missing_data_test = pd.concat([total_test, percent_test], axis=1, keys=['Total_test', 'Percent_test'])

missing_data_test.head(20)

#train

delete=missing_data[missing_data['Percent']>15]

delete1=delete.reset_index()

delete2=delete1['index'].values.tolist()

print(delete2)

#print(df['BsmtCond'])

#print((delete.values.tolist()))

df.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'],axis=1,inplace=True)

#test

delete_test=missing_data_test[missing_data_test['Percent_test']>15]

delete1_test=delete_test.reset_index()

delete2_test=delete1_test['index'].values.tolist()

print(delete2_test)

#print(df['BsmtCond'])

#print((delete.values.tolist()))

#df_test.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'],axis=1,inplace=True)
#test

df_test.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'],axis=1,inplace=True)
#train

sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
#test

sns.heatmap(df_test.isnull(),yticklabels=False,cbar=False)
#train

total = df.isnull().sum().sort_values(ascending=False)

percent = ((df.isnull().sum()/1460)*100).sort_values(ascending=True)



missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



total=total.reset_index()

j=[]

k=[]

for i in total['index']:

    j.append(df[i].dtypes)

    k.append(i)

    #print(df[i].dtypes)



df1 = pd.DataFrame(list(zip(k, j)),columns =['name','type'])  

df1=df1.set_index('name')

#df2 = pd.DataFrame(k,columns =['name'])              

missing_data1 = pd.concat([missing_data,df1], axis=1)

missing_data1.head(20)
#test

total_test = df_test.isnull().sum().sort_values(ascending=False)

percent_test = ((df_test.isnull().sum()/1460)*100).sort_values(ascending=True)



missing_data_test = pd.concat([total_test, percent_test], axis=1, keys=['Total_test', 'Percent_test'])



total_test=total_test.reset_index()

j_test=[]

k_test=[]

for i in total_test['index']:

    j_test.append(df_test[i].dtypes)

    k_test.append(i)

    #print(df[i].dtypes)



df1_test = pd.DataFrame(list(zip(k_test, j_test)),columns =['name','type'])  

df1_test=df1_test.set_index('name')

#df2 = pd.DataFrame(k,columns =['name'])              

missing_data1_test = pd.concat([missing_data_test,df1_test], axis=1)

missing_data1_test.head(75)
#train

df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])

df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])

df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])

df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])

df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])

df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])

df['BsmtFinType1']=df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])

df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])

df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])

df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])

df['Electrical']=df['Electrical'].fillna(df['Electrical'].mode()[0])

df['GarageYrBlt']=df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean())

df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mean())
df
#test

df_test['GarageType']=df_test['GarageType'].fillna(df_test['GarageType'].mode()[0])

df_test['GarageFinish']=df_test['GarageFinish'].fillna(df_test['GarageFinish'].mode()[0])

df_test['GarageCond']=df_test['GarageCond'].fillna(df_test['GarageCond'].mode()[0])

df_test['GarageQual']=df_test['GarageQual'].fillna(df_test['GarageQual'].mode()[0])

df_test['BsmtExposure']=df_test['BsmtExposure'].fillna(df_test['BsmtExposure'].mode()[0])

df_test['BsmtFinType2']=df_test['BsmtFinType2'].fillna(df_test['BsmtFinType2'].mode()[0])

df_test['BsmtFinType1']=df_test['BsmtFinType1'].fillna(df_test['BsmtFinType1'].mode()[0])

df_test['BsmtCond']=df_test['BsmtCond'].fillna(df_test['BsmtCond'].mode()[0])

df_test['BsmtQual']=df_test['BsmtQual'].fillna(df_test['BsmtQual'].mode()[0])

df_test['MasVnrType']=df_test['MasVnrType'].fillna(df_test['MasVnrType'].mode()[0])

df_test['Electrical']=df_test['Electrical'].fillna(df_test['Electrical'].mode()[0])

df_test['GarageYrBlt']=df_test['GarageYrBlt'].fillna(df_test['GarageYrBlt'].mean())

df_test['MasVnrArea']=df_test['MasVnrArea'].fillna(df_test['MasVnrArea'].mean())

df_test['BsmtFullBath']=df_test['BsmtFullBath'].fillna(df_test['BsmtFullBath'].mode()[0])
#train

sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
#test

df_test.dropna(inplace=True)



sns.heatmap(df_test.isnull(),yticklabels=False,cbar=False)
df_train_heat=df[['LotArea','MSZoning','Utilities','Condition1','BldgType','OverallQual',

     'OverallCond','YearBuilt','ExterQual','ExterCond','GrLivArea','GarageArea','PoolArea','SaleType','SaleCondition','SalePrice']]



type(df_train_heat)

corrmat = df_train_heat.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.set(font_scale=1.25)

sns.heatmap(corrmat,annot=True);
df_test.to_csv('df_testc.csv')

df.to_csv('df_c.csv')
df_testc  = pd.read_csv('df_testc.csv')



dfc  = pd.read_csv('df_c.csv')

dfc=dfc.drop(['Unnamed: 0','Id'], axis = 1) 

df_testc=df_testc.drop(['Unnamed: 0','Id'], axis = 1) 

dfc['SalePrice']

#dfc
final_df=pd.concat([dfc,df_testc],axis=0)

final_df
def category_onehot_multcols(multcolumns):

    df_final=final_df

    i=0

    for fields in multcolumns:

        

        print('fields',fields)

        df1=pd.get_dummies(final_df[fields],drop_first=True)

        print('df1',df1)

        final_df.drop([fields],axis=1,inplace=True)

        if i==0:

            df_final=df1.copy()

            print('df_final',df_final)

        else:

            

            df_final=pd.concat([df_final,df1],axis=1)

        i=i+1

       

        

    df_final=pd.concat([final_df,df_final],axis=1)

    print('df_final',df_final)

    print('final_df',final_df)

        

    return df_final
columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',

         'Condition2','BldgType','Condition1','HouseStyle','SaleType','SaleCondition','ExterCond',

         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',

        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',

         'CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']

final_df=category_onehot_multcols(columns)
final_df.shape
final_df =final_df.loc[:,~final_df.columns.duplicated()]

final_df
df_Train=final_df.iloc[:1459,:]

df_Test=final_df.iloc[1460:,:]
df_Test['SalePrice']
df_Test.drop(['SalePrice'],axis=1,inplace=True) 
y_train=df_Train['SalePrice']

X_train=df_Train.drop(['SalePrice'],axis=1)

#xgboost

import xgboost

classifier=xgboost.XGBRegressor()

regressor=xgboost.XGBRegressor()

from sklearn.model_selection import RandomizedSearchCV

booster=['gbtree','gblinear']

base_score=[0.25,0.5,0.75,1]

n_estimators = [100, 500, 900, 1100, 1500]

max_depth = [2, 3, 5, 10, 15]

booster=['gbtree','gblinear']

learning_rate=[0.05,0.1,0.15,0.20]

min_child_weight=[1,2,3,4]



hyperparameter_grid = {

    'n_estimators': n_estimators,

    'max_depth':max_depth,

    'learning_rate':learning_rate,

    'min_child_weight':min_child_weight,

    'booster':booster,

    'base_score':base_score

    }

random_cv = RandomizedSearchCV(estimator=regressor,

                                param_distributions=hyperparameter_grid,

                                cv=5, n_iter=50,

                                scoring = 'neg_mean_absolute_error',n_jobs = 4,

                                verbose = 5, 

                                return_train_score = True,

                                random_state=42)

random_cv.fit(X_train,y_train)

random_cv.best_estimator_

random_cv.best_params_

regressor=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

             importance_type='gain', interaction_constraints=None,

             learning_rate=0.1, max_delta_step=0, max_depth=2,

             min_child_weight=1, missing=None, monotone_constraints=None,

              n_jobs=0, num_parallel_tree=1,

             objective='reg:squarederror', random_state=0, reg_alpha=0,

             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,

             validate_parameters=False, verbosity=None,n_estimators= 900,

            )
regressor.fit(X_train,y_train)

y_pred=regressor.predict(df_Test)
y_pred1=pd.DataFrame(y_pred,columns=['SalePrice'])

type(y_pred1)
sub_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

datasets=pd.concat([sub_df['Id'],y_pred1],axis=1)

datasets.to_csv('sample_submission.csv',index=False)