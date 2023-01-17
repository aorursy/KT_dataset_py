import warnings

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge

from functools import partial

print(os.listdir("../input"))
# Loading training data

base_df = pd.read_csv("../input/train.csv")

base_df.shape
df = pd.read_csv("../input/train.csv")

df.groupby("ExterQual")["SalePrice"].mean().plot.bar()
def transform_rating_to_number(label):

    return {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}.get(label, 0)



df = pd.read_csv("../input/train.csv")

df["ExterQual"] = df["ExterQual"].apply(transform_rating_to_number)

df.groupby("ExterQual")["SalePrice"].mean().plot.bar()
def transform_rating_to_number(label):

    mapping = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}

    return mapping.get(label, 0)





def transform_to_number_bsmt_exposure(label):

    mapping = {"Gd": 5, "Av": 4, "Mn": 3, "No": 2, "NA": 1}

    return mapping.get(label, 0)





def transform_to_boolean(value_for_ok, feature):

    return 1 if value_for_ok == feature else 0





def transform_to_garage_finish(label):

    mapping = {"Fin": 3, "RFn": 2, "Unf": 1, "NA": 0}

    return mapping.get(label, 0)





def transform_paved_drive(label):

    mapping = {"Y": 2, "P": 1, "N": 0}

    return mapping.get(label, 0)





def transform_sale_type(label):

    mapping = {"New": 2, "WD": 1, "NAN": 0}

    return mapping.get(label, 0)





def transform_sale_condition(label):

    mapping = {"Partial": 3, "Normal": 2, "Abnorml": 1, "NAN": 0}

    return mapping.get(label, 0)





def transform_data(_df):

    _df["SaleCondition"] = _df["SaleCondition"].apply(transform_sale_condition)

    _df["FireplaceQu"] = _df["FireplaceQu"].apply(transform_rating_to_number)

    _df["KitchenQual"] = _df["KitchenQual"].apply(transform_rating_to_number)

    _df["HeatingQC"] = _df["HeatingQC"].apply(transform_rating_to_number)

    _df["BsmtQual"] = _df["BsmtQual"].apply(transform_rating_to_number)

    _df["BsmtCond"] = _df["BsmtCond"].apply(transform_rating_to_number)

    _df["ExterQual"] = _df["ExterQual"].apply(transform_rating_to_number)

    _df["BsmtExposure"] = _df["BsmtExposure"].apply(transform_to_number_bsmt_exposure)

    _df["GarageFinish"] = _df["GarageFinish"].apply(transform_to_garage_finish)

    _df["Foundation"] = _df["Foundation"].apply(partial(transform_to_boolean, "PConc"))

    _df["CentralAir"] = _df["CentralAir"].apply(partial(transform_to_boolean, "Y"))

    _df["PavedDrive"] = _df["PavedDrive"].apply(transform_paved_drive)

    _df["SaleType"] = _df["SaleType"].apply(transform_sale_type)



    _df["GarageCond"] = _df["GarageCond"].apply(transform_rating_to_number)

    _df["GarageQual"] = _df["GarageQual"].apply(transform_rating_to_number)

    _df["ExterCond"] = _df["ExterCond"].apply(transform_rating_to_number)



    return _df
plt.scatter(df["LotFrontage"], df["SalePrice"])

plt.scatter(df[df["LotFrontage"] > 250]["LotFrontage"], df[df["LotFrontage"] > 250]["SalePrice"], color="red")
def filter_numerical_data_for_training(_df):

    _df = _df[_df.Id != 496]  # OpenPorchSF outlier



    _df = _df[_df["LotFrontage"] < 250]

    _df = _df[_df["LotArea"] < 100000]

    _df = _df[_df["TotalBsmtSF"] < 3100]



    _df = _df[_df["GarageArea"] < 1200]

    _df = _df[_df["MasVnrArea"] < 1300]

    _df = _df[_df["EnclosedPorch"] < 500]



    return _df





def transform_numerical_data(_df):

    _df["TotalIndorArea"] = _df["1stFlrSF"] + _df["2ndFlrSF"]

    return _df
def remove_unused_features(_df):

    _DELETE = ["Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope",

               "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",

               "Exterior1st", "Exterior2nd", "MasVnrType", "BsmtFinType1", "BsmtFinType2",

               "Heating", "Electrical", "PoolQC", "Fence", "MiscFeature", "Functional"]

    _DELETE_2 = ["GarageType", "MSZoning", "Neighborhood"]



    _NUM_DELETE = ["MSSubClass", "BsmtUnfSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",

                   "BsmtFullBath", "BsmtHalfBath", "KitchenAbvGr", "3SsnPorch", "ScreenPorch",

                   "PoolArea", "MiscVal", "MoSold", "YrSold"]

    _NUM_DELETE_2 = ["BsmtFinSF1", "BsmtFinSF2", "HalfBath", "BedroomAbvGr", "WoodDeckSF"]



    _df = _df.drop(_DELETE + _DELETE_2 + _NUM_DELETE + _NUM_DELETE_2, 1)

    return _df
df = pd.read_csv("../input/train.csv")

df.SalePrice.hist(bins=100)
np.log(df.SalePrice).hist(bins=100)
# Wrapper function to apply cross validation

def evaluate_model_cv(_df, _sale_price, _model):

    scores = -1 * cross_val_score(_model, _df, _sale_price, cv=5, scoring="neg_mean_squared_error")

    return scores.mean()
#Read data set and apply feature engineering

df = pd.read_csv("../input/train.csv")

# fill missing values with 0

df = df.fillna(0)



df = transform_data(df)

df = transform_numerical_data(df)

df = filter_numerical_data_for_training(df)

df = remove_unused_features(df)



sale_price = np.log(df["SalePrice"])

df = df.drop(["Id", "SalePrice"], 1)
print("LinearRegression|{:.10f}".format(evaluate_model_cv(df, sale_price, LinearRegression())))
alphas = np.linspace(0, 30, num=100)

tuned_parameters = [{'alpha': alphas}]

clf = GridSearchCV(Ridge(), tuned_parameters, cv=5, scoring="neg_mean_squared_error")

clf.fit(df, sale_price)

print("Ridge|best_score:{}".format(-clf.best_score_))

print("Ridge|best_estimator_:{}".format(clf.best_estimator_.alpha))
ridge_final_model = Ridge(alpha=clf.best_estimator_.alpha)

ridge_final_model.fit(df, sale_price)



df_test = pd.read_csv("../input/test.csv")

# Test data indexes

test_indexes = df_test.Id



# Transform testing data 

df_test = transform_data(df_test)

df_test = transform_numerical_data(df_test)

df_test = remove_unused_features(df_test)

df_test = df_test.drop(["Id"], 1)

df_test = df_test.fillna(0)



# get predictions

predictions = ridge_final_model.predict(df_test)

# predictions are made on np.log(SalePrice) so transform it back by using np.exp

predictions = np.exp(predictions)



# Print results into a file

prediction = pd.DataFrame({"SalePrice": predictions})

prediction["Id"] = test_indexes

prediction.to_csv("solution.csv", index=False, columns=["Id", "SalePrice"])

print(prediction.head())