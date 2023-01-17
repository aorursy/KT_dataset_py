# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier

%matplotlib inline

sns.set(style="whitegrid")



import warnings

warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore", category=DeprecationWarning) 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
training = pd.read_csv("../input/train.csv")

testing = pd.read_csv("../input/test.csv")
training.head()
testing.head()
training.shape
testing.shape
training.columns
# Lets find missing values!

training_null = pd.isnull(training).sum()

testing_null = pd.isnull(testing).sum()



missing = pd.concat([training_null, testing_null], axis=1, keys=["Training", "Testing"])
missing_many = missing[missing.sum(axis=1) > 200]  #a lot of missing values

missing_few = missing[(missing.sum(axis=1) > 0) & (missing.sum(axis=1) < 200)]  #not as much missing values

# So I can analyze missing_many.

missing_many
missing_few
## Categorical

missing_categorical= ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]

for i in missing_categorical:

    training[i].fillna("None", inplace=True)

    testing[i].fillna("None", inplace=True)
training_null = pd.isnull(training).sum()

testing_null = pd.isnull(testing).sum()



missing = pd.concat([training_null, testing_null], axis=1, keys=["Training", "Testing"])
missing_many = missing[missing.sum(axis=1) > 200]  #a lot of missing values

missing_few = missing[(missing.sum(axis=1) > 0) & (missing.sum(axis=1) < 200)]  #not as much missing values

# So I can analyze missing_many.

missing_many
training.drop("LotFrontage", axis=1, inplace=True)

testing.drop("LotFrontage", axis=1, inplace=True)
missing_few
## Numeric

from sklearn.preprocessing import Imputer



imputer = Imputer(strategy="median")
training["GarageYrBlt"].fillna(training["GarageYrBlt"].median(), inplace=True)

testing["GarageYrBlt"].fillna(testing["GarageYrBlt"].median(), inplace=True)

training["MasVnrArea"].fillna(training["MasVnrArea"].median(), inplace=True)

testing["MasVnrArea"].fillna(testing["MasVnrArea"].median(), inplace=True)

training["MasVnrType"].fillna("None", inplace=True)

testing["MasVnrType"].fillna("None", inplace=True)
#I should convert num_train and num_test to a list to make it easier to work with

num_train = training.dtypes[(training.dtypes == float) | (training.dtypes == int)] #categorical values are type object

num_test = training.dtypes[(training.dtypes == float) | (training.dtypes == int)] #categorical values are type object

numerical_values_train = list(num_train.index)

numerical_values_test = list(num_test.index)

print(numerical_values_train)
fill_num = []



for i in numerical_values_train:

    if i in list(missing_few.index):

        fill_num.append(i)

print(fill_num)
for i in fill_num:

    training[i].fillna(training[i].median(), inplace=True)

    testing[i].fillna(testing[i].median(), inplace=True)
#I should convert cat_train and cat_test to a list to make it easier to work with

cat_train = training.dtypes[training.dtypes == object] #categorical values are type object

cat_test = training.dtypes[training.dtypes == object] #categorical values are type object

categorical_values_train = list(cat_train.index)

categorical_values_test = list(cat_test.index)

print(categorical_values_train)
fill_cat = []



for i in categorical_values_train:

    if i in list(missing_few.index):

        fill_cat.append(i)

print(fill_cat)
for i in categorical_values_train:

    feature_set = set(training[i])

    for j in feature_set:

        feature_list = list(feature_set)

        training.loc[training[i] == j, i] = feature_list.index(j)



for i in categorical_values_test:

    feature_set2 = set(testing[i])

    for j in feature_set2:

        feature_list2 = list(feature_set2)

        testing.loc[testing[i] == j, i] = feature_list2.index(j)
training.head()
testing.head()
most_common = ["Electrical", "Exterior1st", "Exterior2nd", "Functional", "KitchenQual", "MSZoning", "SaleType", "Utilities", "MasVnrType"]
training[most_common].head()
training["MasVnrType"].mean()

#training["Electrical"].mean()

#training["Functional"].mean()
training["Electrical"].fillna(1, inplace=True)

training["Exterior1st"].fillna(4, inplace=True)

training["Exterior2nd"].fillna(6, inplace=True)

training["Functional"].fillna(6, inplace=True)

training["KitchenQual"].fillna(2, inplace=True)

training["MSZoning"].fillna(0, inplace=True)

training["SaleType"].fillna(8, inplace=True)

training["Utilities"].fillna(1, inplace=True)

training["MasVnrType"].fillna(2, inplace=True)
testing["Electrical"].fillna(1, inplace=True)

testing["Exterior1st"].fillna(4, inplace=True)

testing["Exterior2nd"].fillna(6, inplace=True)

testing["Functional"].fillna(6, inplace=True)

testing["KitchenQual"].fillna(2, inplace=True)

testing["MSZoning"].fillna(0, inplace=True)

testing["SaleType"].fillna(8, inplace=True)

testing["Utilities"].fillna(1, inplace=True)

testing["MasVnrType"].fillna(2, inplace=True)
training_null = pd.isnull(training).sum()

testing_null = pd.isnull(testing).sum()



null = pd.concat([training_null, testing_null], axis=1, keys=["Training", "Testing"])

null[null.sum(axis=1) > 0]
correlations= training.corr()

correlations = correlations["SalePrice"].sort_values(ascending=False)

correlations
# According to that list, OverallQual, GrLivArea, GarageCars, GarageArea and TotalBsmtSF are the best corr. 

features = correlations.index[0:6]

features
#Correlation map

f,ax= plt.subplots(figsize=(5,5))

sns.heatmap(training.loc[:,features].corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)

plt.show()
# Let's take a look at our target.

sns.distplot(training["SalePrice"])
sns.distplot(np.log(training["SalePrice"]))
training["TransformedPrice"] = np.log(training["SalePrice"])
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split #to create validation data set
x_train = training.drop(["Id", "SalePrice", "TransformedPrice"], axis=1).values

y_train = training["TransformedPrice"].values

x_test = testing.drop("Id", axis=1).values
X_training, X_valid, y_training, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=0) #X_valid and y_valid are the validation sets
rf = RandomForestRegressor()

paremeters_rf = {"n_estimators" : [5, 10, 15, 20], "criterion" : ["mse" , "mae"], "min_samples_split" : [2, 3, 5, 10], 

                 "max_features" : ["auto", "log2"]}

grid_rf = GridSearchCV(rf, paremeters_rf, verbose=1, scoring="r2")

grid_rf.fit(X_training, y_training)



print("Best RandomForestRegressor Model: " + str(grid_rf.best_estimator_))

print("RF Score: " + str(grid_rf.best_score_))
rf = grid_rf.best_estimator_

rf.fit(X_training, y_training)

rf_pred = rf.predict(X_valid)

r2_rf = r2_score(y_valid, rf_pred)

rmse_rf = np.sqrt(mean_squared_error(y_valid, rf_pred))

print("R^2 Score: " + str(r2_rf))

print("RMSE Score: " + str(rmse_rf))
submission_predictions = np.exp(rf.predict(x_test))
submission = pd.DataFrame({

        "Id": testing["Id"],

        "SalePrice": submission_predictions

    })



submission.to_csv("prices.csv", index=False)

print(submission.shape)
submission.head()