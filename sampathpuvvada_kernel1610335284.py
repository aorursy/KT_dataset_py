# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
### Find out the features which are important.

### Matplotlib
import pandas as pd

import sklearn.preprocessing as pre

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.compose import make_column_transformer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import OneHotEncoder





data=pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")

print(data.columns)

# le = pre.LabelEncoder()

# le.fit(data.MSZoning.unique())

# data["MSZoning"]=le.transform(data["MSZoning"])

# le.fit(data.Street.unique())

# data["Street"]=le.transform(data["Street"])

# le.fit(data.LotShape.unique())

# data["LotShape"]=le.transform(data["LotShape"])

# le.fit(data.Utilities.unique())

# data["Utilities"]=le.transform(data["Utilities"])

# le.fit(data.LotConfig.unique())

# data["LotConfig"]=le.transform(data["LotConfig"])

# le.fit(data.Neighborhood.unique())

# data["Neighborhood"]=le.transform(data["Neighborhood"])

# le.fit(data.BldgType.unique())

# data["BldgType"]=le.transform(data["BldgType"])

# le.fit(data.HouseStyle.unique())

# data["HouseStyle"]=le.transform(data["HouseStyle"])

y=data["SalePrice"]
# Following data has null values.

# Column {} has these many nulls {} BsmtQual 37

# Column {} has these many nulls {} BsmtCond 37

# Column {} has these many nulls {} BsmtExposure 38

# Column {} has these many nulls {} BsmtFinType1 37

# Column {} has these many nulls {} FireplaceQu 690

# Column {} has these many nulls {} GarageType 81

# Column {} has these many nulls {} PoolQC 1453

# Column {} has these many nulls {} MiscFeature 1406



#Columns with NaN are 'FireplaceQu','PoolQC','MiscFeature', 


features=["MSSubClass","MSZoning","LotArea","Street","LotShape","Utilities","LotConfig","Neighborhood","BldgType","HouseStyle",

         "OverallQual","OverallCond","YearBuilt","YearRemodAdd",'Condition1','RoofStyle','RoofMatl','Exterior1st','Exterior2nd'

          ,'MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1'

          ,'Heating','CentralAir','KitchenQual','Functional' ,'GarageType','SaleType','SaleCondition']



features_to_transform=['MSZoning','Street','LotShape','Utilities','LotConfig','Neighborhood','Condition1','BldgType'

                     ,'HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond'

                    ,'Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','Heating','CentralAir','KitchenQual','Functional'

                     ,'GarageType','SaleType','SaleCondition']



X=data[features]

for feature in features:

    number_of_nulls =data[feature].isnull().sum()

    if(number_of_nulls>0):

        print('Column {} has {} nulls'.format(feature,number_of_nulls))

        X.drop(columns=feature)

        features_to_transform.remove(feature)



for feature in features_to_transform:

    for value in X[feature]:

        if(value=='NaN'):

            X.drop(columns=feature)

            features_to_transform.remove(feature)

            break

        

column_trans = make_column_transformer(

   (OneHotEncoder(),features_to_transform),

    remainder='passthrough')



print(X[features_to_transform])

column_trans.fit_transform(X[features_to_transform],X)



print(X[0:1])

print(column_trans.fit_transform(X)[0:1])
rf_model = RandomForestRegressor(random_state=1)



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

#print(data.head())



rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))