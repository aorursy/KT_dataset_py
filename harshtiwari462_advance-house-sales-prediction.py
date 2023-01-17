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
df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
pd.set_option("display.max_columns",None)
df.head()

df.columns
df.head()

df.info()
features_na = [features for features in df.columns if df[features].isnull().sum() >1]
len(features_na)
features_na

import matplotlib.pyplot as plt
for feature in features_na:
    
    data = df.copy()
    data[feature] = np.where(data[feature].isnull(),1,0)
    data.groupby(feature)["SalePrice"].median().plot.bar()
    plt.show()
data = df.copy()
num_features = [feature for feature in data.columns if data[feature].dtypes != 'O']


#data[num_features].head()
print(len(num_features))
num_features
data  = df.copy()
yr_features = [feature for feature in data.columns if 'Yr' in feature or "Year" in feature]

yr_features
for feature in yr_features:
    data.groupby(feature)["SalePrice"].median().plot()
    plt.show()
for feature in yr_features:
    data = df.copy()
    if feature != "YrSold":
        data[feature] = data["YrSold"] - df[feature]
        
        
        plt.scatter(data[feature],data["SalePrice"])
        plt.xlabel(feature)
        plt.ylabel("Saleprice")
        plt.show()
data =  df.copy()

cat_features = [feature for feature in df.columns if df[feature].dtypes == 'O']

print(len(cat_features))
cat_features

    
for feature in cat_features:
    data = df.copy()
    data.groupby(feature)["SalePrice"].median().plot.bar()
    plt.xlabel(feature)
    plt.show()
import seaborn as sns

sns.barplot(x = "MSZoning",y = "SalePrice",hue = "Street",data = data)

sns.barplot(x = "GarageCond",y = "SalePrice",hue = "GarageQual",data = data)
sns.barplot(x = "GarageType",y = "SalePrice",hue = "GarageQual",data = data)
sns.barplot(x = "GarageType",y = "SalePrice",hue = "GarageFinish",data = data)
#sns.barplot(x = "GarageType",y = "SalePrice",hue = "FireplaceQu",data = data)
data = df.copy()
dis_features = [feature for feature in num_features if len(data[feature].unique()) < 25 and feature not in yr_features + ["Id"]]
print(len(dis_features))
dis_features
cont_features = [feature for feature in num_features if feature not in dis_features +  yr_features + ["Id"]]
print(len(cont_features))
cont_features

for feature in cont_features:
    data = df.copy()
    data[feature].hist(bins = 25)
    plt.xlabel(feature)
    plt.show() 
    
for feature in cont_features:
    data = df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        plt.scatter(data[feature],data["SalePrice"])
        plt.xlabel(feature)
        plt.ylabel("Saleprice")
        plt.show() 
    
        
        
        

for feature in cont_features:
    data = df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data.boxplot(column=feature)
        
        plt.show()
    
    
 
   
    
# cat feature with Nan values
cat_features_nan = [feature for feature in df.columns if df[feature].isnull().sum() >1 and df[feature].dtypes == "O" ]
len(cat_features_nan)
def cat_feature_na(df,cat_features_nan):
    data = df.copy()
    data[cat_features_nan] = data[cat_features_nan].fillna("MIssing")
    return data 


df = cat_feature_na(df,cat_features_nan)

df[cat_features_nan].isnull().sum()
# numerical var with na values
num_feature_nan = [feature for feature in df.columns if df[feature].isnull().sum() > 1 and df[feature].dtypes  != "O"]
len(num_feature_nan)
for feature in num_feature_nan:
    
    median = df[feature].median()
    df[feature + 'Nan']  = np.where(df[feature].isnull(),1,0)# to keep track of the NAN VALUE
    df[feature].fillna(median,inplace = True)
    
    
df[num_feature_nan].isnull().sum()
    
     
     
    
    
    
   
yr_feat = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']

for feature in yr_feat:
    df[feature] = df["YrSold"] - df[feature]
    
    

     
df.head()
# converting skewed num var into log normal distribution

num_feat=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
for feature in num_feat:
        df[feature] = np.log(df[feature])
        
        
df.head()
    
for feature in cat_features:
    temp=df.groupby(feature)['SalePrice'].count()/len(df)
    temp_df=temp[temp>0.01].index
    df[feature]=np.where(df[feature].isin(temp_df),df[feature],'Rare_var')
df["GarageCars"].dtype
df.head()
for feature in cat_features:
    labels_ordered=df.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    df[feature]=df[feature].map(labels_ordered)
   
df.head()
scale_features = [feature for feature in df.columns if feature not in ["Id","SalePrice"]]
len(scale_features)
scale_features = [feature for feature in df.columns if feature not in ["Id","SalePrice"]]


from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
scale.fit(df[scale_features])
scale.transform(df[scale_features])
# concatinating the "ID" and SalePRice column with the transformed scalefeatures
train_df = pd.concat([df[["Id","SalePrice"]].reset_index(drop = True),
                    pd.DataFrame(scale.transform(df[scale_features]),columns = scale_features)],axis = 1)
train_df.head()
train_df.to_csv("train_csv",index = False)
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
train_df = pd.read_csv("train_csv")
y_train = train_df[["SalePrice"]]
x_train = train_df.drop(["Id","SalePrice"],axis =1)
feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0)) # remember to set the seed, the random state in this function
feature_sel_model.fit(x_train, y_train)
feature_sel_model.get_support()
selected_feat = x_train.columns[(feature_sel_model.get_support())]
selected_feat
x_train = x_train[selected_feat]
x_train.head()
x_train.isnull().sum().any
test_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
test_df.head()
test_df.shape
data = test_df.copy()
num_features = [feature for feature in test_df.columns if test_df[feature].dtypes != 'O']


#data[num_features].head()
print(len(num_features))
num_features
data  = test_df.copy()
yr_features = [feature for feature in test_df.columns if 'Yr' in feature or "Year" in feature]

yr_features
data =  test_df.copy()

cat_features = [feature for feature in test_df.columns if test_df[feature].dtypes == 'O']

print(len(cat_features))
cat_features

data = test_df.copy()
dis_features = [feature for feature in num_features if len(data[feature].unique()) < 25 and feature not in yr_features + ["Id"]]
print(len(dis_features))
dis_features
cont_features = [feature for feature in num_features if feature not in dis_features +  yr_features + ["Id"]]
print(len(cont_features))
cont_features
cat_features_nan = [feature for feature in test_df.columns if test_df[feature].isnull().sum() > 1 and test_df[feature].dtypes == "O" ]
len(cat_features_nan)
def cat_feature_na(test_df,cat_features_nan):
    data = test_df.copy()
    data[cat_features_nan] = data[cat_features_nan].fillna("MIssing")
    return data 


test_df = cat_feature_na(test_df,cat_features_nan)

test_df[cat_features_nan].isnull().sum()
num_feature_nan = [feature for feature in test_df.columns if test_df[feature].isnull().sum() > 1 and test_df[feature].dtypes  != "O"]
len(num_feature_nan)
for feature in num_feature_nan:
    
    median = test_df[feature].median()
    test_df[feature + 'Nan']  = np.where(test_df[feature].isnull(),1,0)# to keep track of the NAN VALUE
    test_df[feature].fillna(median,inplace = True)
    
    
test_df[num_feature_nan].isnull().sum()
yr_feat = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']

for feature in yr_feat:
    test_df[feature] = test_df["YrSold"] - test_df[feature]
    
num_feat=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']
for feature in num_feat:
        test_df[feature] = np.log(test_df[feature])
        
        
test_df.head()
    
for feature in cat_features:
    temp=test_df.groupby(feature).count()/len(test_df)
    temp_df=temp[temp>0.01].index
    test_df[feature]=np.where(test_df[feature].isin(temp_df),test_df[feature],'Rare_var')
for feature in cat_features:
    labels_ordered=test_df.groupby([feature]).mean().sort_values(by = ["Id"]).index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    test_df[feature]=test_df[feature].map(labels_ordered)
scale_features = [feature for feature in test_df.columns if feature not in ["Id"]]
len(scale_features)
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
scale.fit(test_df[scale_features])
scale.transform(test_df[scale_features])
test_df = pd.concat([test_df[["Id"]].reset_index(drop = True),
                    pd.DataFrame(scale.transform(test_df[scale_features]),columns = scale_features)],axis = 1)
test_df.head()
test_df.describe()
x_test= test_df.drop(["Id"],axis =1)
x_test = test_df[['MSSubClass', 'MSZoning', 'Neighborhood', 'OverallQual', 'YearRemodAdd',
       'RoofStyle', 'BsmtQual', 'BsmtExposure', 'HeatingQC', 'CentralAir',
       '1stFlrSF', 'GrLivArea', 'BsmtFullBath', 'KitchenQual', 'Fireplaces',
       'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageCars', 'PavedDrive',
       'SaleCondition']]
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
lr.coef_
lr.intercept_
x_test["GarageCars"].index[x_test["GarageCars"].apply(np.isnan)]
x_test.drop([1116],inplace = True)
sales_prediction = lr.predict(x_test)
sales_prediction
final = pd.concat([test_df[["Id"]].reset_index(drop = True),pd.DataFrame((sales_prediction),columns = ["Sales_price"])],axis = 1)
final.head()
final.to_csv("test_csv",index = False)

