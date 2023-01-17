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
df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df.head()
dfTest=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df.head()
#Concatting Train and Test data for Data cleansing

df["flag"]="0"
dfTest["flag"]="1"
FinalHouse_Data=pd.concat([df,dfTest])
FinalHouse_Data.head()
FinalHouse_Data.isnull().sum()

missing_value_count_coulmn=FinalHouse_Data.isnull().sum()
print(missing_value_count_coulmn[missing_value_count_coulmn>0])
#missingValues=pd.DataFrame(data=missing_value_count_coulmn[missing_value_count_coulmn>0],columns=['A','B'])
missingValues=pd.DataFrame(missing_value_count_coulmn[missing_value_count_coulmn>0])
list(missingValues.index)
# Removing LotFrontage feature as too many null values. Other null values can be extrapolated as most of them are NA
#FinalHouse_Data=FinalHouse_Data.drop(['LotFrontage'],axis=1,inplace=True)
print(FinalHouse_Data.columns.values)
FinalHouse_Data.shape
Ordinal_categorical=["MSSubClass","Street","Alley","LotShape","LandContour","Utilities","LotConfig","LandSlope","Condition1"
                    ,"BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st","MasVnrType","ExterQual","ExterCond","Foundation"
                    ,"BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Heating","HeatingQC","CentralAir",
                    "Electrical","KitchenQual","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond","PavedDrive",
                    "PoolQC","Fence","MSZoning","MiscFeature","SaleType","SaleCondition"]
#Nominal_categorical=["MSZoning","MiscFeature","SaleType","SaleCondition"]
def Simplenullremoval(df,column):
    for col in column:
        if((df[col].dtypes=='int64') | (df[col].dtypes=='int32')):
            df[col].fillna(0,inplace=True)
        df[col].fillna("Not Applicable",inplace=True)
    print(df[column].head())
    return df

 
        
ParsesimplenullData=['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','Fence','MiscFeature','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond','PoolQC']
HouseData_nullcheck=Simplenullremoval(FinalHouse_Data,ParsesimplenullData)
FinalHouse_Data[ParsesimplenullData]=HouseData_nullcheck[ParsesimplenullData]
FinalHouse_Data.shape

missing_value_count_coulmn=FinalHouse_Data.isnull().sum()
print(missing_value_count_coulmn[missing_value_count_coulmn>0])
#missingValues=pd.DataFrame(data=missing_value_count_coulmn[missing_value_count_coulmn>0],columns=['A','B'])
missingValues=pd.DataFrame(missing_value_count_coulmn[missing_value_count_coulmn>0])
list(missingValues.index)
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(strategy='most_frequent')
nullCategoryValues=['MSZoning','Utilities','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','Electrical','KitchenQual','Functional','SaleType']
#HouseData_nullcategory=pd.DataFrame(imputer.fit_transform(FinalHouse_Data[nullCategoryValues]))
HouseData_nullcategory=pd.DataFrame(imputer.fit_transform(FinalHouse_Data[nullCategoryValues]))
HouseData_nullcategory.columns=df[nullCategoryValues].columns

HouseData_nullcategory.head()
FinalHouse_Data[nullCategoryValues]=HouseData_nullcategory
FinalHouse_Data.head()

missing_value_count_coulmn=FinalHouse_Data.isnull().sum()
print(missing_value_count_coulmn[missing_value_count_coulmn>0])
#missingValues=pd.DataFrame(data=missing_value_count_coulmn[missing_value_count_coulmn>0],columns=['A','B'])
missingValues=pd.DataFrame(missing_value_count_coulmn[missing_value_count_coulmn>0])
list(missingValues.index)
Meanimputer=SimpleImputer(strategy='mean')
nullCategoryValues=['SalePrice']
#HouseData_nullcategory=pd.DataFrame(imputer.fit_transform(FinalHouse_Data[nullCategoryValues]))
HouseData_nullcategorySale=pd.DataFrame(Meanimputer.fit_transform(FinalHouse_Data[nullCategoryValues]))
HouseData_nullcategorySale.columns=df[nullCategoryValues].columns

HouseData_nullcategorySale.head()
FinalHouse_Data[nullCategoryValues]=HouseData_nullcategorySale
FinalHouse_Data.head()
#Y=FinalHouse_Data['SalePrice']
# Removing the flag, saleprice and id columns from the features matrix
cols = [col for col in FinalHouse_Data.columns if col in ['flag','SalePrice','Id']]
Y=FinalHouse_Data[cols]
X=FinalHouse_Data.drop(['LotFrontage','Id','SalePrice'],axis=1,inplace=True)
FinalHouse_Data.head()
# Selecting the train part of the dataset by making flag=0
X_train=FinalHouse_Data[FinalHouse_Data["flag"]=='0']
X_test=FinalHouse_Data[FinalHouse_Data["flag"]=='1']
Y.head()
Y_train=Y[Y["flag"]=='0']
y_train=Y_train.drop(['flag','Id'],axis=1,inplace=True)
Y_test=Y[Y["flag"]=='1']
y_test=Y_test.drop(['flag','Id'],axis=1,inplace=True)

print(Y_test)


from sklearn.preprocessing import LabelEncoder

def transformCategorical(df,columns):
    label_encoder=LabelEncoder()
    for col in columns:
        if df[col].dtype=='object':
            df[col]=label_encoder.fit_transform(df[col].astype(str))
    return df[columns]
        
X_train[X_train.columns.values]=transformCategorical(X_train,X_train.columns.values)
X_train.head()

def transformCategoricalTest(df,columns):
    label_encoder=LabelEncoder()
    for col in columns:
        if df[col].dtype=='object':
            df[col]=label_encoder.fit_transform(df[col].astype(str))
    return df[columns]

X_test[X_test.columns.values]=transformCategoricalTest(X_test,X_test.columns.values)
X_test.head()
for col in (X_train.columns.values):
    if X_train[col].dtype=='object':
        print (col)
from sklearn.linear_model import LinearRegression
# Instantiating linear regressor
lm=LinearRegression()
# Fitting the model on the training data
lm.fit(X_train,Y_train)

# Prediction on the training data
predictions_linearregressor_traindata=lm.predict(X_test)
print(predictions_linearregressor_traindata.shape)
print(Y_test.shape)
predicted=pd.DataFrame(data=predictions_linearregressor_traindata)
#dff = pd.DataFrame({'Actual': Y_test, 'Predicted': predictions_linearregressor_traindata})

dff=pd.concat([Y_test,predicted],axis=1)
dff

