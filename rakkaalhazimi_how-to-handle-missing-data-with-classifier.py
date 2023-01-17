import numpy as np

import pandas as pd
# Fetch the kaggle file path

import os



for dirpath, dirname, filename in os.walk("/kaggle/input/"):

    dirname = dirname

    dirpath = dirpath

    print(dirpath, dirname, filename)
dirpath
# Define train data, train target and test data

dtrain = pd.read_csv(dirpath + "/train.csv")

dtarget = dtrain["SalePrice"]

dtrain.drop(columns="SalePrice", inplace=True)

dtest = pd.read_csv(dirpath + "/test.csv")



# Concatenate both train and test data

dwhole = pd.concat([dtrain, dtest], axis=0)



dwhole
# Check missing value

print("There are {} columns containing missing values".format(dwhole.isnull().any().sum()))



null = dwhole.isnull().sum()

null = null[null > 0].sort_values(ascending=False)

null
# Missing values dtypes

print( dwhole[null.index].dtypes.value_counts() )

print()
# Missing values counts for object

for col in dwhole[null.index].select_dtypes("O").columns:

    print("Missing value: ", dwhole[col].isnull().sum())

    print(dwhole[col].value_counts(), end="\n\n")
# Missing values counts for integer and float

for col in dwhole[null.index].select_dtypes("float").columns:

    print("Missing value: ", dwhole[col].isnull().sum())

    print(dwhole[col].value_counts(), end="\n\n")
# Record the missing values type and values needed

# Based on data description

objectMiss = {"none" : ["PoolQC" ,"MiscFeature" ,"Alley", "Fence", "FireplaceQu", "GarageFinish", "GarageQual", 

                        "GarageCond", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", 

                        "BsmtFinType1", "MasVnrType"],

             "notNone": ["MSZoning", "Utilities", "Functional", "Exterior2nd", "Exterior1st", "SaleType", 

                         "Electrical", "KitchenQual"]

}





floatMiss = {"zero" : ["LotFrontage", "MasVnrArea", "BsmtFullBath", "BsmtHalfBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "GarageCars", 

                     "GarageArea", "TotalBsmtSF"],

             "notZero" : ["GarageYrBlt"]

}
# Validate total na columns

total = 0

for key in objectMiss.keys():

    total += len(objectMiss[key])



for key in floatMiss.keys():

    total += len(floatMiss[key])

    

print("Total missing: ", total)
# Fill None and Zero values

dwhole[objectMiss["none"]] = dwhole[objectMiss["none"]].fillna("None")

dwhole[floatMiss["zero"]] = dwhole[floatMiss["zero"]].fillna(0.)
# Fill Not Zero values

dwhole.drop(columns=floatMiss["notZero"], inplace=True)
# Specify notNone index

nanIndex = dwhole[dwhole[objectMiss["notNone"]].isnull().any(axis=1)][objectMiss["notNone"]]

nanIndex
# Specify value counts

for col in dwhole[objectMiss["notNone"]].columns:

    print(dwhole[col].value_counts())

    print()
# Copy the dataframe for safety

dwhole_copy = dwhole.copy()

nanCl = nanIndex.copy()



# Targets are object dtypes excluding the nan index

targets = dwhole_copy.drop(nanIndex.index)[nanIndex.columns]



# Features are float dtypes excluding the nan index

features = dwhole_copy.drop(nanIndex.index).select_dtypes(["int64", "float64"]).drop(columns="Id")



# predictions are float dtypes including the nan FROM categorical index

predictions = dwhole_copy.select_dtypes(["int64", "float64"]).drop(columns="Id").iloc[nanIndex.index]



# Here the prediction features, it has same rows with the missing values

predictions
# Our missing values before

nanCl
# Use classifier to fill NA values

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier



label = LabelEncoder()

forest = RandomForestClassifier(random_state=0)
import re

# Iterate throught columns

for col in nanCl.columns:

    # Fit label

    label.fit(targets[col].value_counts().index)

    encoded = label.transform(targets[col].values)

    # Fit classifier

    forest.fit(features, encoded)

    # Prediction index

    predict = nanIndex[nanIndex[col].isnull()].index 

    

    for pre in predict:

        # Predict features from index

        result = forest.predict(predictions.loc[pre].to_numpy().reshape(1, -1))

        # Inverse transform integer to label

        decoded = label.inverse_transform(result)

        # Product from transform is np.array, convert it to string 

        decoded = np.array2string(decoded)

        # Use regex to extract the words

        regex = re.findall("[A-Za-z]+", decoded)

        # Insert this value to dataframe

        dwhole_copy.loc[pre, col] = regex[0] # regex product is list, slice it to get the string

        print("Change %d to %s" % (result, regex[0]))
# Double check the data frame missing values

dwhole_copy.isnull().sum().sum()
# Save to csv

dwhole_copy.drop(columns="Id", inplace=True)

dwhole_copy.to_csv("full_features.csv", index=False)
dwhole_copy = pd.read_csv("full_features.csv")
# Check our new data

dwhole_copy
# Convert categorical data to 0 or 1 values

def ohe(df):

    dfCopy = df.copy()

    dfCopy = pd.get_dummies(df, drop_first=True)

    return dfCopy



oheFeatures = ohe(dwhole_copy)

oheFeatures
# Convert categorical data based on frequency of the values

def countE(df):

    dfCopy = df.copy()

    train = df.iloc[:1460]

    test = df.iloc[1460:]

    

    

    for col in train.select_dtypes("O"):

        countMap = train[col].value_counts().to_dict()

        dfCopy[col] = dfCopy[col].map(countMap)

    

    return dfCopy

        

ceFeature = countE(dwhole_copy)

ceFeature
# Convert categorical data to ordered values with respect to target mean

def targetE(df):

    dfCopy = df.copy()

    train = df.iloc[:1460]

    target = dtarget

    test = df.iloc[1460:]

    

    full = pd.concat([train, target], axis=1)

    

    for col in full.select_dtypes("O"):

        orderedMap = full.groupby([col])["SalePrice"].mean().sort_values().index

        

        orderedMap = {k:i for i, k in enumerate(orderedMap)}

        dfCopy[col] = dfCopy[col].map(orderedMap)

    

    return dfCopy

        

targetFeature = targetE(dwhole_copy)

targetFeature
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.linear_model import Ridge, Lasso, ElasticNet

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_log_error



estimators = [GradientBoostingRegressor, RandomForestRegressor, Ridge, Lasso, XGBRegressor]
def estimatorEval(estimators, features, target):

    X = features.iloc[:1460].copy()

    Y = target.copy()

    

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

    

    for estimator in estimators:

        print(estimator.__name__)

        est = estimator(random_state=11).fit(X_train, y_train)

        print("Training Score :", est.score(X_train, y_train))

        print("Test Score :", est.score(X_test, y_test))

        print("RMSLE root squared :", np.sqrt(mean_squared_log_error(abs(y_test), abs(est.predict(X_test)))))

        print()

    

    return est

    
# One hot encoding

estimatorEval(estimators, oheFeatures, dtarget)
# Count encoding

estimatorEval(estimators, ceFeature, dtarget)
# Ordered categorical encoding

estimatorEval(estimators, targetFeature, dtarget)
# Gradien boost is better

new_estimators = [GradientBoostingRegressor]



# Create new feature with log, polynomial and both

targetFlog = pd.concat([targetFeature, np.log1p(targetFeature.add_suffix("_log"))], axis=1)

targetPoly = pd.concat([targetFeature, np.power(targetFeature.add_suffix("_poly"), 2)], axis=1)

targetPF = pd.concat([targetFlog, np.power(targetFeature.add_suffix("_poly"), 2)], axis=1)
estimatorEval(new_estimators, targetFlog, dtarget)
estimatorEval(new_estimators, targetPoly, dtarget)
estimatorEval(new_estimators, targetPF, dtarget)
gboost = estimatorEval([GradientBoostingRegressor], targetPoly, dtarget)
# Submit the results

perceive = gboost.predict(targetPoly.iloc[1460:])

submit = pd.DataFrame({"Id": np.arange(1461, 2920), "SalePrice": perceive})

submit.to_csv("submission5b.csv", index=False)