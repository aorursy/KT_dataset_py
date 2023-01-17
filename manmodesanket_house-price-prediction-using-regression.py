import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv('../input/train.csv', header=0)

test = pd.read_csv('../input/test.csv', header=0)
train.head()
#As told by author

train = train[train['GrLivArea']<4000]
y = train['SalePrice'].values
#Utilities-all the values are same in this field

train.drop(["Utilities"],axis=1,inplace=True)

test.drop(["Utilities"],axis=1,inplace=True)
X_train = pd.DataFrame(train.iloc[:,:-1])
X_train_full = pd.concat([X_train,test],axis=0)
#seperating numerical and categorical features

categoricals = X_train_full.select_dtypes(include=['object'])

numericals = X_train_full.select_dtypes(exclude=['object'])
categoricals.isnull().sum()
#missing data columns

columns_absent = [

        'GarageCond', 'GarageType', 'GarageQual', 'GarageFinish',

        'Fence', 'PoolQC', 'Alley', 'FireplaceQu', 'MiscFeature',

        'BsmtFinType1', 'BsmtCond', 'BsmtFinType2', 'BsmtQual','BsmtExposure'

    ]
categoricals[columns_absent] = categoricals[columns_absent].fillna("Absent")



categoricals['MSZoning'] = categoricals['MSZoning'].fillna(categoricals['MSZoning'].mode()[0])

categoricals['Electrical'] = categoricals['Electrical'].fillna(categoricals['Electrical'].mode()[0])

categoricals['KitchenQual'] = categoricals['KitchenQual'].fillna(categoricals['KitchenQual'].mode()[0])

categoricals['Functional'] = categoricals['Functional'].fillna(categoricals['Functional'].mode()[0])

categoricals['SaleType'] = categoricals['SaleType'].fillna(categoricals['SaleType'].mode()[0])

categoricals['Exterior1st'] = categoricals['Exterior1st'].fillna(categoricals['Exterior1st'].mode()[0])

categoricals['Exterior2nd'] = categoricals['Exterior2nd'].fillna(categoricals['Exterior2nd'].mode()[0])



categoricals['MasVnrType'] = categoricals['MasVnrType'].fillna("None")

categoricals.isnull().sum()
missing = [col for col in numericals.columns if numericals[col].isnull().any()]



num_missing = list(set(numericals.columns).intersection(set(missing)))



columns_zero = ['BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'BsmtFinSF2',

                    'BsmtUnfSF', 'BsmtFinSF1', 'GarageArea', 'TotalBsmtSF', 'MasVnrArea']
numericals[columns_zero] = numericals[columns_zero].fillna(0)



values_mssubclass = []

for i in numericals.MSSubClass.unique():

    values_mssubclass.append("SC"+str(i))

    

values_yearbuilt = []

for i in numericals.YearBuilt.unique():

    values_yearbuilt.append("Y"+str(i))

    

values_yearremodadd = []

for i in numericals.YearRemodAdd.unique():

    values_yearremodadd.append("Y"+str(i))



values_mssubclass = dict(zip(numericals.MSSubClass.unique(), values_mssubclass))

values_yearbuilt = dict(zip(numericals.YearBuilt.unique(), values_yearbuilt))

values_yearremodadd = dict(zip(numericals.YearRemodAdd.unique(), values_yearremodadd))



numericals = numericals.replace(

        {

            'MSSubClass' : values_mssubclass,

            'MoSold' : {1 : 'Jan', 2 : 'Feb', 3 : 'Mar', 4 : 'Apr', 5 : 'May', 6 : 'Jun',

                        7 : 'Jul', 8 : 'Aug', 9 : 'Sep', 10 : 'Oct', 11 : 'Nov', 12 : 'Dec'},

            'YrSold': {2006: 'Y2006', 2007: 'Y2007', 2008: 'Y2008', 2009: 'Y2009', 2010:'Y2010'},

            'YearBuilt': values_yearbuilt,

            'YearRemodAdd': values_yearremodadd,

        }

    )
numericals.isnull().sum()
numericals['LotFrontage'] = numericals['LotFrontage'].fillna(0)
numericals['GarageYrBlt'] = numericals['GarageYrBlt'].fillna(numericals['YearBuilt'])
numericals.isnull().sum()
final_data = pd.concat([numericals,categoricals],axis=1)
cor = final_data.corr()
print(cor)
plt.figure(figsize=[30,15])

sns.heatmap(cor,annot=True)

plt.show()
drop_columns = ['OverallCond' , 'BsmtFinSF2' , 'LowQualFinSF' , 'BsmtHalfBath' , 

                'GarageArea', 'TotRmsAbvGrd' , '3SsnPorch' , 'PoolArea' , 'MiscVal'] 



final_data.drop(drop_columns,axis=1,inplace=True)
final_data.shape
final_data.drop(['Alley'],axis=1,inplace=True)
final_data = pd.get_dummies(final_data,drop_first=True)
print(final_data.shape)

print(train.shape)
X_train = final_data.iloc[:1456,:].values

X_test = final_data.iloc[1456:,:].values
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y)  
y_pred = regressor.predict(X_test)
y_pred = pd.DataFrame(y_pred)



y_pred.columns = ["SalePrice"] 



id = pd.read_csv('../input/train.csv', header=0)





pred = pd.concat([id["Id"],y_pred["SalePrice"]],axis=1)

pred["Id"] = pred["Id"] + 1460

pred.drop(pred.index[[1459]],inplace=True)

pred.to_csv("submission_final.csv",index=False)
pred.shape