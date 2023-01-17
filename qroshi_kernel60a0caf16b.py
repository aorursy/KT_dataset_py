# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
with open('/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt', "r") as file:
    print(*(file.readlines()))
data.head(10)
data.info()
data.describe()
data["SalePrice"].describe()
data["SalePrice"].hist()
plt.scatter(data["SalePrice"], data["Id"])
data_prepared = data.copy()
data_prepared.drop("Id", axis=1, inplace=True)
data_prepared.groupby("MSSubClass")["SalePrice"].mean().sort_values(ascending=False)
#so that later code will not treat this as a numerical feature
data_prepared["MSSubClass"] = data_prepared["MSSubClass"].astype(np.str)
dtypes = data_prepared.dtypes
object_features = dtypes[dtypes == np.object].index.values
num_features = dtypes[(dtypes == np.int64) | (dtypes == np.float64)].index.values
print(object_features)
print(len(object_features))
print(num_features)
print(len(num_features))
#delete outliers
data_prepared.query("SalePrice < 500000 and SalePrice > 50000", inplace=True)
data_prepared.reset_index(inplace=True)
data_prepared.describe()
data_prepared["SalePrice"].describe()
#handle NaN values that have meaning 
na_values = {"BsmtCond": "None", "BsmtFinType1": "None", "BsmtExposure": "None",
             "BsmtFinType2": "None", "BsmtQual": "None", "FireplaceQu": "None", 
             "GarageType": "None", "GarageFinish": "None", "GarageQual": "None",  
             "GarageCond": "None", "PoolQC": "None", "Fence": "None", 
             "MiscFeature": "None", "MasVnrType": "None", "Alley": "None",
             "GarageYrBlt": data_prepared["GarageYrBlt"].min()-1, "MasVnrArea": 0}  

data_prepared.fillna(value=na_values, inplace=True)
na_counts = data_prepared.isna().sum()
na_counts[na_counts != 0]
features_with_na = na_counts[na_counts != 0].index.values
features_with_na
data_prepared["LotFrontage"].hist()
data_prepared["LotFrontage"].median()
for x in features_with_na: 
    if x in num_features: data_prepared[x].fillna(data_prepared[x].mean(), inplace=True)
    elif x in object_features: data_prepared[x].fillna(data_prepared[x].mode()[0], inplace=True)
na_counts = data_prepared.isna().sum()
na_counts[na_counts != 0]
data_prepared["LotFrontage"].describe()
for x in object_features: print(data.groupby(x)["SalePrice"].mean().sort_values(ascending=False))
import matplotlib.pyplot as plt 
data["Utilities"].hist()
data["MSZoning"].hist()
ordinal_categories = {x : data_prepared.groupby(x)["SalePrice"].mean().sort_values() for x in object_features}
ordinal_categories["Neighborhood"]
data_prepared["GarageType"].hist()
full_cat_to_ord_dict = {x : {} for x in ordinal_categories}
full_cat_to_ord_dict
for x in ordinal_categories: 
    len_choices = len(ordinal_categories[x]) 
    for y in range(len_choices): full_cat_to_ord_dict[x][ordinal_categories[x].index[y]] = y#ordinal_categories[x][y]
full_cat_to_ord_dict["MasVnrType"]
full_cat_to_ord_dict["SaleCondition"]
for x in full_cat_to_ord_dict: 
    if "None" in full_cat_to_ord_dict[x].keys() and full_cat_to_ord_dict[x]["None"] != 0: 
        del full_cat_to_ord_dict[x]["None"]
        newsorted = sorted(full_cat_to_ord_dict[x], key=full_cat_to_ord_dict[x].get)
        full_cat_to_ord_dict[x] = {newsorted[x]: x+1 for x in range(len(newsorted))}
        full_cat_to_ord_dict[x]["None"] = 0
        print(full_cat_to_ord_dict[x])
        print(x)
data["Utilities"]
for x in ordinal_categories: 
    data_prepared[x] = data_prepared[x].map(full_cat_to_ord_dict[x])
data_prepared.loc[ data_prepared["Neighborhood"] <= 4, "Neighborhood"]                                           = 0  
data_prepared.loc[(data_prepared["Neighborhood"] >=  5) & (data_prepared["Neighborhood"] <=  9), "Neighborhood"] = 1  
data_prepared.loc[(data_prepared["Neighborhood"] >= 10) & (data_prepared["Neighborhood"] <= 14), "Neighborhood"] = 2  
data_prepared.loc[(data_prepared["Neighborhood"] >= 15) & (data_prepared["Neighborhood"] <= 19), "Neighborhood"] = 3  
data_prepared.loc[(data_prepared["Neighborhood"] >= 20) & (data_prepared["Neighborhood"] <= 24), "Neighborhood"] = 4  
data_prepared.info()
data["GarageType"].hist()
data_prepared["GarageType"].hist()
data["Neighborhood"].hist()
data_prepared["Neighborhood"].hist()
data_prepared.head()
data_prepared["GarageYrBlt"] -= data_prepared["GarageYrBlt"].min() 
data_prepared["YrSold"] -= data_prepared["YrSold"].min()
data.query("SalePrice < 500000 and SalePrice > 50000")["YrSold"].hist()
data_prepared["YrSold"].hist()
data.query("SalePrice < 500000 and SalePrice > 50000")["GarageYrBlt"].hist()
data_prepared["GarageYrBlt"].hist()
data_prepared.drop("index", axis=1, inplace=True)
data_prepared
garage_year_min = data["GarageYrBlt"].min()-1
year_sold_min = data["YrSold"].min()

def preprocess_data(data):
    data_prepared = data.copy()
    data_prepared.drop("Id", axis=1, inplace=True)
    
    #so that later code will not treat this as a numerical feature
    data_prepared["MSSubClass"] = data_prepared["MSSubClass"].astype(np.str)
    
    dtypes = data_prepared.dtypes
    object_features = dtypes[dtypes == np.object].index.values
    num_features = dtypes[(dtypes == np.int64) | (dtypes == np.float64)].index.values
    
    #handle NaN values that have meaning 
    data_prepared.fillna(value=na_values, inplace=True)
    
    #fill missing NaN values
    na_counts = data_prepared.isna().sum()
    features_with_na = na_counts[na_counts != 0].index.values
    for x in features_with_na: 
        if x in num_features: data_prepared[x].fillna(data_prepared[x].mean(), inplace=True)
        elif x in object_features: data_prepared[x].fillna(data_prepared[x].mode()[0], inplace=True)
    
    #converting cat features to ordinals 
    for x in object_features: 
        data_prepared[x] = data_prepared[x].map(full_cat_to_ord_dict[x])
    
    #converting Neighborhood numerical feature to ordinal    
    data_prepared.loc[ data_prepared["Neighborhood"] <= 4, "Neighborhood"]                                       = 0  
    data_prepared.loc[(data_prepared["Neighborhood"] >=  5) & (data_prepared["Neighborhood"] <=  9), "Neighborhood"] = 1  
    data_prepared.loc[(data_prepared["Neighborhood"] >= 10) & (data_prepared["Neighborhood"] <= 14), "Neighborhood"] = 2  
    data_prepared.loc[(data_prepared["Neighborhood"] >= 15) & (data_prepared["Neighborhood"] <= 19), "Neighborhood"] = 3  
    data_prepared.loc[(data_prepared["Neighborhood"] >= 20) & (data_prepared["Neighborhood"] <= 24), "Neighborhood"] = 4 
    
    #handling year feautures 
    data_prepared["GarageYrBlt"] -= garage_year_min
    data_prepared["YrSold"] -= year_sold_min 
    
    return data_prepared.astype(np.float64)
X_train = data.copy()
y_train = X_train["SalePrice"]
X_train.drop("SalePrice", axis=1, inplace=True)
X_train = preprocess_data(X_train) 
data.head()
X_train.head()
from sklearn.ensemble import RandomForestRegressor 

forest_reg = RandomForestRegressor() 
forest_reg.fit(X_train, y_train)  
forest_reg.score(X_train, y_train)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
print(-cross_val_score(forest_reg, X_train, y_train, cv=3, scoring="neg_mean_squared_log_error"))
forest_reg = RandomForestRegressor() 
forest_reg.fit(X_train, y_train)
X_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
X_test_prepared = preprocess_data(X_test).fillna(2)
predictions = forest_reg.predict(X_test_prepared)
result = pd.DataFrame({"Id": X_test["Id"], 
                       "SalePrice": predictions})
result.to_csv("submission.csv", index=False)