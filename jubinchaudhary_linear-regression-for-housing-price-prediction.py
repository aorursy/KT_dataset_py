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
df_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
df_train.head()
train_Id = df_train["Id"]
test_Id = df_test["Id"]
df_train.describe()
# 37+1(y= saleprice) numerical data columns and 43 categorical data columns 
df_test.head()
df_test.describe()
sns.heatmap(df_train.isnull())
sns.heatmap(df_test.isnull())
mat=df_train.corr()

fig,ax= plt.subplots(figsize=(30,30))
sns.heatmap(mat,annot = True, annot_kws={'size' : 12})
# from the heat plot shown above, we analysed by salePrice has maximum dependency on 'GrLivArea','OverallQual','GarageCars','GarageArea'. 
abc= df_train[['GrLivArea','OverallQual','GarageCars','GarageArea', 'SalePrice' ]]
sns.pairplot(abc)

df_train.info()
# 43 object data type means 43 categorical data type columns
df_test.info()
df_train.isnull().sum().sort_values(ascending = False)[0:20]
df_test.isnull().sum().sort_values(ascending = False)[0:35]
#deleting those columns which have more than 50% NaN values
#as those columns are same for both test and train datas
# why garageYrBlt
list_drop=["PoolQC","MiscFeature","Alley","Fence","GarageYrBlt"]

for col in list_drop:
    del df_train[col]
    del df_test[col]
    

df_train.shape
df_train.isnull().sum().sort_values(ascending=False)[0:15]
df_test.isnull().sum().sort_values(ascending=False)[0:30]
df_train.LotFrontage.value_counts(dropna=False)
df_train.LotFrontage.fillna(df_train.LotFrontage.mean(),inplace=True)
df_test.LotFrontage.fillna(df_test.LotFrontage.mean(),inplace=True)
df_train.shape

list_fill_train=["BsmtCond", "BsmtQual", "GarageType", "GarageCond", "GarageFinish",
                 "GarageQual","MasVnrType","BsmtFinType2","BsmtExposure","FireplaceQu","MasVnrArea"]

for j in list_fill_train:
    #df_train[j].fillna(df_train[j].mode(),inplace=True).
    # wrong way to do it.
    # mode() return a tuple : mode value and freuency of that value , therefore using [0] gives access to mode value.
    df_train[j] = df_train[j].fillna(df_train[j].mode()[0])
    df_test[j] = df_test[j].fillna(df_train[j].mode()[0])

df_train.shape

print(df_train.isnull().sum().sort_values(ascending=False)[0:5])
print(df_test.isnull().sum().sort_values(ascending=False)[0:20])
# BsmtFinType1    37
#Electrical       has to be deleted i.E 1460-38 = 1422 
# total =38 rows 
df_train.dropna(inplace=True)
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
# one time .any returns true or false for each column , if any 1 single vaule of column is na , it will return false against 
#that column. if we apply .any again, it will return if any sinfle column has an entry ,as false....i.e if the entire dataframe 
# is has any na value.



#joining data sets
df_final=pd.concat([df_train,df_test],axis=0)
df_final.shape

df_final.info()
# 39 objecttype columns which has to be converted into numerical data so that lr can be applied.
columns = ['MSZoning', 'Street','LotShape', 'LandContour', 'Utilities',
           'LotConfig', 'LandSlope','Neighborhood', 'Condition1', 'Condition2',
           'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
           'Exterior2nd', 'MasVnrType','ExterQual', 'ExterCond', 'Foundation',
           'BsmtQual', 'BsmtCond','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
           'Heating', 'HeatingQC', 'CentralAir', 'Electrical','KitchenQual',
           'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
           'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
def One_hot_encoding(columns):
    final_df=df_final
    i=0 # means MSZoning
    for fields in columns:
        #get dummies function numericalize(1s, 0s) the caterogical column and stored into df1
        df1=pd.get_dummies(df_final[fields],drop_first=True)
        
        df_final.drop([fields],axis=1,inplace=True)
        if i==0: #this will be executed only for MSZoning
            final_df=df1.copy() # the new numerical data(1s ,0s) MSZoning is being copied into final_df  
        else:           
            final_df=pd.concat([final_df,df1],axis=1)
        i=i+1
       
     # before execution of next statement, df_final has no categorical column and final_df has all the corresponding
        # (to categorical column ) numericalised(1s, 0s) column. 
    final_df=pd.concat([df_final,final_df],axis=1)
    
        
    return final_df
main_df=df_train.copy()
df_final.head()

df_final = One_hot_encoding(columns)
df_final.shape
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
X_train, X_test, Y_train, Y_test = train_test_split(x_train_final, y_train_final)
print(X_train.shape,X_test.shape)
print(Y_train.shape,Y_test.shape)
##model building
linear_reg=LinearRegression()
linear_reg.fit(X_train,Y_train)
Y_pred = linear_reg.predict(X_test)
print("R-Squared Value for Training Set: {:.3f}".format(linear_reg.score(X_train,Y_train)))
print("R-Squared Value for Test Set: {:.3f}".format(linear_reg.score(X_test,Y_test)))
print(r2_score(Y_test, Y_pred))
y_pred_test=linear_reg.predict(df_test_m)
pred_df = pd.DataFrame(y_pred_test, columns=['SalePrice'])
test_id_df = pd.DataFrame(test_Id, columns=['Id'])
submission = pd.concat([test_id_df, pred_df], axis=1)
submission.head()

submission.to_csv('submission.csv', index=False)



