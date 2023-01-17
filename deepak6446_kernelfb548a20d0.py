# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train =  pd.read_csv('../input/train.csv')

train.head()
train.shape
if('Id' in train.columns):

    del train['Id']

train.head()
# lets count the data distribution

import seaborn as sns

sns.countplot(train["MSSubClass"])
# lets see how the data is distributed

sns.distplot(train['SalePrice'])
# lets check the skewness of data to see its distribution across median.

train['SalePrice'].skew()
# Identifies the type of dwelling involved in the sale.	

train['MSSubClass'].unique()
##### Finding and filling null values #####



# count number of null values in data

def find_null_values(data):    

    null_values = data.isnull().sum().sort_values(ascending=False)

    null_values = null_values[null_values != 0]

    print("shape: ", train.shape)

    print(null_values)



find_null_values(train)
# PoolQC, MiscFeature, Alley and Fence data is of no use to us as almost all values are null 

def delete_null_values(data):

    try: 

        del data["PoolQC"]

        del data["MiscFeature"]

        del data["Alley"]

        del data["Fence"]

    except Exception as e:

        print("execption occured: ", e)

    return data



train = delete_null_values(train)

find_null_values(train)
# Fireplace quality

train["FireplaceQu"].unique()
# fill null values with NA meaning no fireplace in dwelling 

train["FireplaceQu"].fillna("NA", inplace=True)

train["FireplaceQu"].unique()
# LotFrontage: Linear feet of street connected to property

train["LotFrontage"].fillna(0, inplace=True)
# GarageCond: Garage condition

# lets replace null values for garbage with TA (Typical/Average)

train["GarageCond"].fillna("NA", inplace=True)

train["GarageType"].fillna("NA", inplace=True)

train["GarageYrBlt"].fillna("NA", inplace=True)

train["GarageFinish"].fillna("NA", inplace=True)

train["GarageQual"].fillna("NA", inplace=True)
print(train["BsmtExposure"].unique())

train["BsmtExposure"].fillna("No", inplace=True)
# Rating of basement finished area (if multiple types)

print(train["BsmtFinType2"].unique())

train["BsmtFinType2"].fillna("NA", inplace=True)
print(train["BsmtFinType1"].unique())

train["BsmtFinType1"].fillna("NA", inplace=True)
print(train["BsmtCond"].unique())

train["BsmtCond"].fillna("NA", inplace=True)
print(train["BsmtQual"].unique())

train["BsmtQual"].fillna("NA", inplace=True)
sns.distplot(train["MasVnrArea"].dropna())
train["Electrical"].unique()
sns.countplot(train["Electrical"])
train["Electrical"].fillna("Mix", inplace=True)
# Masonry veneer type: type of wall

# data is not distributed so lets replace null values with None

print(train["MasVnrType"].unique())

train["MasVnrType"].fillna("None", inplace=True)

train["MasVnrArea"].fillna(0, inplace=True)
# lets check if we still have any null values in our dataset

find_null_values(train)
pd.set_option("display.max_columns", None)

train.head()
def get_unique_count(data, column):

    print(data[column].value_counts())

get_unique_count(train, "3SsnPorch")
sns.countplot(x="HalfBath", data=train) 
# lets convert categorical data using get dummies

def convert_categorical(data, column):

    return pd.get_dummies(data, columns=column)
# import matplotlib.pyplot as plt

#     fig, ax = plt.subplots(len(columns), 1)

#     [sns.countplot(x=columns[i], data=train, ax = ax[i]) for i in range(0, len(columns))]



# plot_count_plot(["BsmtHalfBath", "3SsnPorch"])
sns.countplot(x="FullBath", data=train) 
sns.countplot(x="FullBath", data=train) 
sns.countplot(x="BsmtFinSF2", data=train) 
# count number of values in GarageYrBlt contaning null values

train["GarageYrBlt"].str.count("NA").sum()
# age_array = dframe[dframe["Age"]!=np.nan]["Age"]



# dframe["Age"].replace(np.nan,age_array.mean())

def replace_null_GarageYrBlt(data):

    garage_array = float(data[data["GarageYrBlt"] !="NA"][["GarageYrBlt"]].mean())

    data.loc[data.GarageYrBlt == "NA", 'GarageYrBlt'] = garage_array

    data['GarageYrBlt'] = data['GarageYrBlt'].astype("float") 

    return data



train = replace_null_GarageYrBlt(train)
# numpy ndarray stores string in object format

categorical_data = [i for i in train.dtypes.keys() if str(train.dtypes[i]) != "int64" and str(train.dtypes[i]) != "float64" ]

categorical_data
for i in categorical_data:

    print(i, train[i].unique())
def get_dummies_categorical(data):

    data = pd.get_dummies(data, columns=categorical_data)

    return data

train = get_dummies_categorical(train)
train.head()
y_train = train["SalePrice"]

train = train.drop("SalePrice", axis=1)
# lets try to normalize our dataset



from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()



train_array = train.values

train_scaled = min_max_scaler.fit_transform(train_array)



train = pd.DataFrame(train_scaled, columns=train.columns)
x_train = train.values

y_train = y_train.values
print(type(x_train))

print(type(y_train))
from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(x_train, y_train, test_size = 0.1, random_state = 10)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score, GridSearchCV
import pickle

# parameters for GridSearchCV

# cv: Determines the cross-validation splitting strategy

# scoring: statergy that is used to measure the performance of diff models 

# verbose: controls the messages displayed, the higher the more messages are displayed

# n_jobs: -1 specify to use all processes



# getting the best paramter for your model using GridSearchCV

# load the model from disk

filename = 'model1.sav'

model = None

try:

    model = pickle.load(open(filename, 'rb'))

except Exception as e:

    print("exception raised: ", e)

if (model == None):

    gsc = GridSearchCV(

            estimator=RandomForestRegressor(),

            param_grid={

                'max_depth': range(6,20),

                'n_estimators': (10, 20, 50, 100, 1000),

            },

            cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)



    grid_result = gsc.fit(xTrain, yTrain)

    best_params = grid_result.best_params_

    

    print("best params: ",best_params) # lets print best params



    model = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"], random_state=False, verbose=False)

    model.fit(xTrain, yTrain)

    

    # save the model to disk

    pickle.dump(model, open(filename, 'wb'))

    # Perform K-Fold CV
scores = cross_val_score(model, xTrain, yTrain, cv=10, scoring='neg_mean_squared_error')
print(scores)
yPred = model.predict(xTest)

from sklearn import metrics



print('Mean Absolute Error:', metrics.mean_absolute_error(yTest, yPred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yTest, yPred)))
ids = None

def create_test_data():

    

    test =  pd.read_csv('../input/test.csv')

    print("test shape: ", test.shape)

    if('Id' in test.columns):

        global ids

        ids = test['Id']

        del test['Id']

    test = delete_null_values(test)

    

    list_fill_na = ("FireplaceQu" ,"GarageCond" ,"GarageType" ,"GarageYrBlt" ,"GarageFinish" ,"GarageQual" ,"BsmtFinType2" ,"BsmtFinType1" ,"BsmtCond", "BsmtQual")

    [test[i].fillna("NA", inplace=True) for i in list_fill_na]

    

    test["LotFrontage"].fillna(0, inplace=True)

    test["BsmtExposure"].fillna("No", inplace=True)

    test["Electrical"].fillna("Mix", inplace=True)

    test["MasVnrType"].fillna("None", inplace=True)

    test["MasVnrArea"].fillna(0, inplace=True)

    

    test = replace_null_GarageYrBlt(test)

    test = get_dummies_categorical(test)

    

    test["BsmtHalfBath"].fillna(test["BsmtHalfBath"].mean(), inplace=True)

    

    list_fill_na = ("BsmtHalfBath" , "BsmtFullBath", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "GarageArea", "BsmtFinSF1", "BsmtFinSF2")

    [test[i].fillna(test[i].mean(), inplace=True) for i in list_fill_na]

    

    return test
x_test = create_test_data()
sns.countplot(x_test["BsmtHalfBath"])
find_null_values(x_test)
print("test features: ", x_test.shape)

print("train features: ", x_train.shape)
# Get missing columns in the training test

missing_cols = set( train.columns ) - set( x_test.columns )

# Add a missing column in test set with default value equal to 0

for c in missing_cols:

    x_test[c] = 0

# Ensure the order of column in the test set is in the same order than in train set

x_test = x_test[train.columns]
print("test features: ", x_test.shape)

print("train features: ", x_train.shape)
test_scaled = min_max_scaler.fit_transform(x_test.values)

y_pred = model.predict(test_scaled)
type(y_pred)
type(ids.values)
df = pd.DataFrame({'Id':ids.values, 'SalePrice':y_pred})
df.head()
df.to_csv("submission.csv", index=None)

df = pd.read_csv("submission.csv")
df.head()