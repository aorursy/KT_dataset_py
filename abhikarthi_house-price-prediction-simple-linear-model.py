# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import skew

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.linear_model import LinearRegression

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



#Data importing

train_data = pd.read_csv("../input/train.csv")

test_data= pd.read_csv("../input/test.csv")

merged_dataset = pd.concat([train_data,test_data], axis=0)

#Check for the count of null values

total = merged_dataset.isnull().sum().sort_values(ascending=False)

percent = (merged_dataset.isnull().sum()/merged_dataset.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

print(missing_data)
#Identifying the variables which are correlated more than 50% with target variable

corrmat = merged_dataset.corr()["SalePrice"]

corrmat = corrmat.sort_values(axis = 0 , ascending = False)

corrmat[corrmat > 0.50]
## Handling the missing values for all the variables

#Numerical variables

merged_dataset.select_dtypes(include=['int64','float64']).columns



#Treating the null values for variable Alley

#sns.countplot(x = 'Alley' , data = merged_dataset)

merged_dataset['Alley'].fillna('None',inplace = True)



#Replacing the Null categorical variables of Bsmtx with None

merged_dataset['BsmtQual'].fillna(value = 'None' , inplace = True)

merged_dataset['BsmtCond'].fillna(value = 'None' , inplace = True)

merged_dataset['BsmtExposure'].fillna(value = 'None' , inplace = True)

merged_dataset['BsmtFinType1'].fillna(value = 'None' , inplace = True)

merged_dataset['BsmtFinType2'].fillna(value = 'None' , inplace = True)



#Replacing the numerical variables of Bsmtx with 0

merged_dataset['BsmtFinSF1'].fillna(value = 0 , inplace = True)

merged_dataset['BsmtFinSF2'].fillna(value = 0 , inplace = True)

merged_dataset['BsmtFullBath'].fillna(value = 0 , inplace = True)

merged_dataset['BsmtHalfBath'].fillna(value = 0 , inplace = True)

merged_dataset['BsmtUnfSF'].fillna(value = 0 , inplace = True)

merged_dataset['TotalBsmtSF'].fillna(value = 0 , inplace = True)



#Since SBrkr is most common for the house, replace the null value Electrical variable with common value

merged_dataset['Electrical'].fillna(value = 'SBrkr' , inplace = True)



#Replacing the FireplaceQu missing value with none

merged_dataset['FireplaceQu'].fillna(value = 'None' , inplace =  True)



#Assuming these houses are not having garage, replacing categorical variables with None and numerical variables with 0

merged_dataset['GarageType'].fillna(value = 'None' , inplace = True)

merged_dataset['GarageYrBlt'].fillna(value = 'None' , inplace = True)

merged_dataset['GarageFinish'].fillna(value = 'None' , inplace = True)

merged_dataset['GarageQual'].fillna(value = 'None' , inplace = True)

merged_dataset['GarageCond'].fillna(value = 'None' , inplace = True)

merged_dataset['GarageArea'].fillna(value = 0 , inplace = True)

merged_dataset['GarageCars'].fillna(value = 0 , inplace = True)



#PoolQC has above 95% of the null values, seems these houses doesnot have pool area, so replacing the categorical variable with None

merged_dataset['PoolQC'].fillna(value = 'None' , inplace = True)



#Treating the NA values of lot frontage with 0, seems house doesn't have lot frontage area

merged_dataset['LotFrontage'].fillna(value = 0, inplace = True)



#Treating Misc feature variables with None for categorical and 0 for numeric

merged_dataset['MiscFeature'].fillna(value='None', inplace = True)

merged_dataset['Exterior1st'].fillna(value= 'None', inplace = True)

merged_dataset['Exterior2nd'].fillna(value= 'None', inplace = True)

merged_dataset['Functional'].fillna(value= 'None', inplace = True)

merged_dataset['KitchenQual'].fillna(value = 'None' , inplace = True)

merged_dataset['MSZoning'].fillna(value = 'None' , inplace = True)

merged_dataset['SaleType'].fillna(value = 'None' , inplace = True)

merged_dataset['Utilities'].fillna(value = 'None' , inplace = True)

merged_dataset["MasVnrType"].fillna(value = 'None', inplace=True)

merged_dataset["MasVnrArea"].fillna(value = 0, inplace =True)

merged_dataset["Fence"].fillna(value = 'None', inplace = True)
#Skewness will affect the performance of the regression model. In order the overcome this, log transformation on the numeric variables will be beneficial

merged_dataset["SalePrice"] = np.log1p(merged_dataset["SalePrice"])



numeric_feats = merged_dataset.dtypes[merged_dataset.dtypes != "object"].index

skewed_feats = merged_dataset[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



merged_dataset[skewed_feats] = np.log1p(merged_dataset[skewed_feats])
#Creating label encoder with the categorical variables

labelEnc=LabelEncoder()



cat_vars=['Alley', 'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',

       'BsmtFinType2', 'BsmtQual', 'CentralAir', 'Condition1', 'Condition2',

       'Electrical', 'ExterCond', 'ExterQual', 'Exterior1st', 'Exterior2nd',

       'Fence', 'FireplaceQu', 'Foundation', 'Functional', 'GarageCond',

       'GarageFinish', 'GarageQual', 'GarageType', 'Heating', 'HeatingQC',

       'HouseStyle', 'KitchenQual', 'LandContour', 'LandSlope', 'LotConfig',

       'LotShape', 'MSZoning', 'MasVnrType', 'MiscFeature', 'Neighborhood',

       'PavedDrive', 'PoolQC', 'RoofMatl', 'RoofStyle', 'SaleCondition',

       'SaleType', 'Street', 'Utilities']



for col in cat_vars:

    merged_dataset[col]=labelEnc.fit_transform(merged_dataset[col])
# Year Columns ---- 'GarageYrBlt','YearBuilt','YearRemodAdd', 'YrSold'



merged_dataset['GarageYrBlt'].replace('None' , 100, inplace = True)



bins = [10,1960, 1980, 2000, 2017]



group_names = ['VeryOld', 'Old', 'Okay', 'New'] #Grouping into categories

merged_dataset['GarageYrBlt'] = pd.cut((merged_dataset['GarageYrBlt']), bins, labels=group_names)

merged_dataset['GarageYrBlt'].fillna('VeryOld', inplace = True)

merged_dataset['YearBuilt'] = pd.cut((merged_dataset['YearBuilt']), bins, labels=group_names)

merged_dataset['YearRemodAdd'] = pd.cut((merged_dataset['YearRemodAdd']), bins, labels=group_names)

merged_dataset['YrSold'] = pd.cut((merged_dataset['YrSold']), bins, labels=group_names)
labelEnc=LabelEncoder()



cat_vars=['GarageYrBlt','YearBuilt','YearRemodAdd', 'YrSold']



for col in cat_vars:

    merged_dataset[col]=labelEnc.fit_transform(merged_dataset[col])
#Spliting the dataset into train and test dataset

new_train = merged_dataset[:1460]

X_train = new_train.drop('SalePrice',axis=1)

y_train = new_train['SalePrice']
new_test = merged_dataset[1460:]

X_test = new_test.drop('SalePrice',axis=1)
lr = LinearRegression().fit(X_train,y_train)

prediction = np.expm1(lr.predict(X_test))

print(prediction)
submission = pd.DataFrame({

        "Id": X_test["Id"],

        "SalePrice": prediction

    })



submission.to_csv("HousePrice.csv", index=False)