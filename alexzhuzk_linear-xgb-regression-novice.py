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
#------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from xgboost import XGBRegressor
#------------------------------------------------------------------------------
#load dataset
train = pd.read_csv("../input/train.csv")

#save and drop train id
train_id = train["Id"]
train.drop(columns='Id',inplace=True)

#select object columns
obj_col = train.columns[train.dtypes == 'object'].values

#select non object columns
num_col = train.columns[train.dtypes != 'object'].values

#replace null value in obj columns with None
train[obj_col] = train[obj_col].fillna('None')

#replace null value in numeric columns with 0
train[num_col] = train[num_col].fillna(0)
#Encode ordinal features
ordinal_features = ["ExterQual","ExterCond","BsmtQual","BsmtCond","BsmtExposure",
                    "BsmtFinType1","BsmtFinType2","HeatingQC","Electrical","KitchenQual",
                    "FireplaceQu","GarageQual","GarageCond","PoolQC"]

ExterQual_map = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"None":0}
ExterCond_map = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1}
BsmtQual_map = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"None":0}
BsmtCond_map = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"None":0}
BsmtExposure_map = {"Gd":5,"Av":4,"Mn":3,"No":2,"None":1}
BsmtFinType1_map = {"GLQ":6,"ALQ":5,"BLQ":4,"Rec":3,"LwQ":2,"Unf":1,"None":0}
BsmtFinType2_map = {"GLQ":6,"ALQ":5,"BLQ":4,"Rec":3,"LwQ":2,"Unf":1,"None":0}
HeatingQC_map = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1}
Electrical_map = {"SBrkr":5,"FuseA":4,"FuseF":3,"FuseP":2,"Mix":1,"None":0}
KitchenQual_map = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"None":0}
FireplaceQu_map = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"None":0}
GarageQual_map = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"None":0}
GarageCond_map = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"None":0}
PoolQC_map = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"None":1}

train["ExterQual"] = train["ExterQual"].map(ExterQual_map)
train["ExterCond"] = train["ExterCond"].map(ExterCond_map)
train["BsmtQual"] = train["BsmtQual"].map(BsmtQual_map)
train["BsmtCond"] = train["BsmtCond"].map(BsmtCond_map)
train["BsmtExposure"] = train["BsmtExposure"].map(BsmtExposure_map)
train["BsmtFinType1"] = train["BsmtFinType1"].map(BsmtFinType1_map)
train["BsmtFinType2"] = train["BsmtFinType2"].map(BsmtFinType2_map)
train["HeatingQC"] = train["HeatingQC"].map(HeatingQC_map)
train["Electrical"] = train["Electrical"].map(Electrical_map)
train["KitchenQual"] = train["KitchenQual"].map(KitchenQual_map)
train["FireplaceQu"] = train["FireplaceQu"].map(FireplaceQu_map)
train["GarageQual"] = train["GarageQual"].map(GarageQual_map)
train["GarageCond"] = train["GarageCond"].map(GarageCond_map)
train["PoolQC"] = train["PoolQC"].map(PoolQC_map)
# Encode nominal features
nominal_features = [x for x in obj_col if x not in ordinal_features]

# Transfer object to int
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

# for loop nominal feature column
for _ in train[nominal_features].columns:
    #fit and transform each column and assign to itself
    train[_] = labelencoder.fit_transform(train[_])

# Get dummy variable for nominal features
train = pd.get_dummies(train,columns=nominal_features,drop_first=True)

# Check if any null values
train.isnull().any().sum()
#------------------------------------------------------------------------------
# Split data to X(features)  and y(target)
# X should be in matrix form and y shoud be in array form
X = train.drop(columns="SalePrice").values
y = train["SalePrice"].values
#------------------------------------------------------------------------------
# Split data to train dataset and test dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler

train_scaler = StandardScaler()
X_train_scaler = train_scaler.fit_transform(X_train)

test_scaler = StandardScaler()
X_test_scaler = test_scaler.fit_transform(X_test)
#------------------------------------------------------------------------------
# Feature Importance
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators = 100,random_state=1,n_jobs=1)
forest.fit(X_train_scaler,y_train)

# Grid Search - 1
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators':[10,100],
               'min_samples_split':[2,4],
               'min_samples_leaf':[1,2]}]
    
grid_search = GridSearchCV(estimator = forest,
                           param_grid = parameters,
                           cv = 5,
                           n_jobs = -1)

grid_search = grid_search.fit(X_train_scaler,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

forest = RandomForestRegressor(n_estimators = 100,
                               min_samples_leaf = 1,
                               min_samples_split = 4,
                               random_state = 1,
                               n_jobs = 1)

forest.fit(X_train_scaler,y_train)
#------------------------------------------------------------------------------
# Feature Selection
from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(forest,threshold = 0.0005,prefit=True)
X_selected  = sfm.transform(X_train_scaler)
#------------------------------------------------------------------------------
# Fit into linear regression model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_selected,y_train)
y_pred = reg.predict(sfm.transform(X_test_scaler))
np.sqrt(((y_pred - y_test)**2).sum())
#------------------------------------------------------------------------------
# Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = reg,X = X_selected, y = y_train,cv=10)
accuracies.mean(),accuracies.std()
#------------------------------------------------------------------------------
# Fit into XGB linear regression model

xgb = XGBRegressor(n_estimators = 100,learning_rate=0.08,gamma=0,subsample=0.75,
                   colsample_bytree = 1,max_depth=7)

xgb.fit(X_selected,y_train)

y_pred_xgb = xgb.predict(sfm.transform(X_test_scaler))
#------------------------------------------------------------------------------
# Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = xgb,X = X_selected, y = y_train,cv=10)
accuracies.mean(),accuracies.std()
#------------------------------------------------------------------------------
#load dataset
test = pd.read_csv("../input/test.csv")
# Save Id
test_id = test["Id"]

# Drop Id column
test.drop(columns=["Id"],inplace=True)

#select object columns
obj_col = test.columns[test.dtypes == 'object'].values

#select non object columns
num_col = test.columns[test.dtypes != 'object'].values

#replace null value in obj columns with None
test[obj_col] = test[obj_col].fillna('None')

#replace null value in numeric columns with 0
test[num_col] = test[num_col].fillna(0)
#------------------------------------------------------------------------------

test["ExterQual"] = test["ExterQual"].map(ExterQual_map)
test["ExterCond"] = test["ExterCond"].map(ExterCond_map)
test["BsmtQual"] = test["BsmtQual"].map(BsmtQual_map)
test["BsmtCond"] = test["BsmtCond"].map(BsmtCond_map)
test["BsmtExposure"] = test["BsmtExposure"].map(BsmtExposure_map)
test["BsmtFinType1"] = test["BsmtFinType1"].map(BsmtFinType1_map)
test["BsmtFinType2"] = test["BsmtFinType2"].map(BsmtFinType2_map)
test["HeatingQC"] = test["HeatingQC"].map(HeatingQC_map)
test["Electrical"] = test["Electrical"].map(Electrical_map)
test["KitchenQual"] = test["KitchenQual"].map(KitchenQual_map)
test["FireplaceQu"] = test["FireplaceQu"].map(FireplaceQu_map)
test["GarageQual"] = test["GarageQual"].map(GarageQual_map)
test["GarageCond"] = test["GarageCond"].map(GarageCond_map)
test["PoolQC"] = test["PoolQC"].map(PoolQC_map)

# Encode nominal features
nominal_features = [x for x in obj_col if x not in ordinal_features]

# Transfer object to int
labelencoder = LabelEncoder()
# for loop nominal feature column
for _ in test[nominal_features].columns:
    #fit and transform each column and assign to itself
    test[_] = labelencoder.fit_transform(test[_])

# Get dummy variable for nominal features
test = pd.get_dummies(test,columns=nominal_features,drop_first=True)

test.isnull().any().sum()
# Get missing columns in the training test
missing_cols = set(train.drop(columns="SalePrice").columns) - set(test.columns)

# Add a missing column in test set with default value equal to 0
for cols in missing_cols:
    test[cols] = 0
    
# Ensure the order of column in the test set is in the same order than in train set
test = test[train.drop(columns="SalePrice").columns]

# Split data to X(features)
test_X = test.values
#------------------------------------------------------------------------------
# Feature Scaling
test_X_scaler = StandardScaler()
test_X_scaler = test_X_scaler.fit_transform(test_X)
#------------------------------------------------------------------------------
# Feature selection
test_X_selected = sfm.transform(test_X_scaler)
#------------------------------------------------------------------------------
# Make prediction
test_y_pred = xgb.predict(test_X_selected)

submission = pd.DataFrame({'Id':test_id,'SalePrice':test_y_pred})

# Save results
submission.to_csv("submission0924.csv",index=False)