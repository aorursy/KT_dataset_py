# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

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
df1=df[['LotArea','MSZoning','Utilities','Condition1','BldgType','OverallQual',
       'OverallCond','YearBuilt','ExterQual','ExterCond','GrLivArea','GarageArea',
       'PoolArea','SaleType','SaleCondition','GrLivArea','GarageArea',
       'TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','SalePrice']].copy()

print(df1)

df.head()
X = df.iloc[:,0:74]  #independent columns
y = df.iloc[:,-1]    #target column i.e price range
X
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
#train
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
#test
df_test['MSZoning']=df_test['MSZoning'].fillna(df_test['MSZoning'].mode()[0])
df_test['Functional']=df_test['Functional'].fillna(df_test['Functional'].mode()[0])
df_test['Utilities']=df_test['Utilities'].fillna(df_test['Utilities'].mode()[0])
df_test['SaleType']=df_test['SaleType'].fillna(df_test['SaleType'].mode()[0])
df_test['Exterior2nd']=df_test['Exterior2nd'].fillna(df_test['Exterior2nd'].mode()[0])
df_test['Exterior1st']=df_test['Exterior1st'].fillna(df_test['Exterior1st'].mode()[0])
df_test['KitchenQual']=df_test['KitchenQual'].fillna(df_test['KitchenQual'].mode()[0])
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
df_test['BsmtHalfBath']=df_test['BsmtHalfBath'].fillna(df_test['BsmtHalfBath'].mean())
df_test['GarageArea']=df_test['GarageArea'].fillna(df_test['GarageArea'].mean())
df_test['BsmtFinSF2']=df_test['BsmtFinSF2'].fillna(df_test['BsmtFinSF2'].mean())
df_test['BsmtUnfSF']=df_test['BsmtUnfSF'].fillna(df_test['BsmtUnfSF'].mean())
df_test['BsmtFinSF1']=df_test['BsmtFinSF1'].fillna(df_test['BsmtFinSF1'].mean())
df_test['GarageCars']=df_test['GarageCars'].fillna(df_test['GarageCars'].mean())
df_test['TotalBsmtSF']=df_test['TotalBsmtSF'].fillna(df_test['TotalBsmtSF'].mean())

#df_test.dropna(inplace=True)
df2=df_test[['LotArea','MSZoning','Utilities','Condition1','BldgType','OverallQual',
       'OverallCond','YearBuilt','ExterQual','ExterCond','GrLivArea','GarageArea',
       'PoolArea','SaleType','SaleCondition','GrLivArea','GarageArea',
       'TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd']].copy()

print(df2.shape)
df2.to_csv('df2_testc.csv')
df1.to_csv('df1_c.csv')
total = df_test.isnull().sum().sort_values(ascending=False)
percent = ((df_test.isnull().sum()/1460)*100).sort_values(ascending=True)

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
#missing_data1.head(20)
df1
#test
sns.heatmap(df2.isnull(),yticklabels=False,cbar=False)
df2_testc= pd.read_csv('df2_testc.csv')

df1_c= pd.read_csv('df1_c.csv')
df1_c=df1_c.drop(['Unnamed: 0'], axis = 1) 
df2_testc=df2_testc.drop(['Unnamed: 0'], axis = 1) 
df1_c
final_df=pd.concat([df1_c,df2_testc],axis=0)
final_df
total = final_df.isnull().sum().sort_values(ascending=False)
percent = ((final_df.isnull().sum()/1460)*100).sort_values(ascending=True)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

total=total.reset_index()
j=[]
k=[]
for i in total['index']:
    j.append(final_df[i].dtypes)
    k.append(i)
    #print(df[i].dtypes)

df3 = pd.DataFrame(list(zip(k, j)),columns =['name','type'])  
df3=df3.set_index('name')
#df2 = pd.DataFrame(k,columns =['name'])              
missing_data1 = pd.concat([missing_data,df3], axis=1)
missing_data1.head(22)
columns=['MSZoning','Utilities','Condition1','BldgType','ExterQual','ExterCond','SaleType','SaleCondition']
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
final_df=category_onehot_multcols(columns)
#df1_c.shape
df2_testc.shape
final_df =final_df.loc[:,~final_df.columns.duplicated()]
final_df.shape
df_Train=final_df.iloc[:1460,:]
df_Test=final_df.iloc[1461:,:]
df_Test.drop(columns=['SalePrice'],axis=1,inplace=True)
df_Test
y_train=df_Train[['SalePrice']]
X_train=df_Train.drop(['SalePrice'],axis=1)
X_train.shape
#y_train.shape
df_Test.shape
#xgboost
import xgboost
classifier=xgboost.XGBRegressor()
regressor=xgboost.XGBRegressor()


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
regressor.fit(X_train,y_train)
y_pred=regressor.predict(df_Test)
y_pred
y_pred1=pd.DataFrame(y_pred,columns=['SalePrice'])
y_pred1
sub_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
datasets=pd.concat([sub_df['Id'],y_pred1],axis=1)
datasets['SalePrice']=datasets['SalePrice'].fillna(datasets['SalePrice'].mean())
datasets
#df2_testc=df2_testc.drop(['Unnamed: 0'], axis = 1) 
datasets.to_csv('sample_submission.csv',index=False)
import pickle
filename = 'finalized_model.pkl'
pickle.dump(regressor, open(filename, 'wb'))
sub_df=pd.read_csv('sample_submission.csv')
sub_df
