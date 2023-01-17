import os

from pathlib import Path

import pandas as pd

import numpy as np
root = Path("../input/home-data-for-ml-course/")
def load_data(root, file_name):

    return pd.read_csv(root/file_name)
os.listdir(root)
train = load_data(root, "train.csv")

test = load_data(root, "test.csv")

submission = load_data(root, "sample_submission.csv")
train.head()
test.head()
train.shape, test.shape
%matplotlib inline

import seaborn as sns

from matplotlib import pyplot as plt

with sns.axes_style(style= "whitegrid"):    

    for dataset in ["train", "test"]:

        fig, ax = plt.subplots(1, 1, figsize=(18, 10))

        bar = eval(dataset).isna().sum().plot(kind = "bar")

        bar.set_title(f"No. of missing values in {dataset}", fontsize = 17)

with sns.axes_style(style = "darkgrid"):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (18, 5))

    dist = sns.distplot(a = train.SalePrice, kde=True, hist=False, rug=True, ax = ax1)

    box = sns.boxplot(x = "SalePrice", data = train, ax = ax2)

#current year = 2020

def feature_eng(df):

    df["HouseAge"] = 2020 - df["YearBuilt"]

    df["BedroomsPerBathrooms"] = df["FullBath"] / df["BedroomAbvGr"]

    df["BedroomsPerRooms"] = (df["FullBath"] + df["KitchenAbvGr"]) / df["BedroomAbvGr"]

    df["TotalSF"] = df["BsmtFinSF1"] + df["BsmtFinSF2"] + df["1stFlrSF"] + df["2ndFlrSF"]

    df["TotalBathrooms"] = df["FullBath"] + (0.5 * df["HalfBath"]) + df["BsmtFullBath"] + (0.5 * df["BsmtHalfBath"])

    df["TotalPorchSF"] = df["OpenPorchSF"] + df["3SsnPorch"] + df["EnclosedPorch"] + df["ScreenPorch"] + df["WoodDeckSF"]

    df["YearBlRm"] = df["YearBuilt"] + df["YearRemodAdd"]



    return df
train = feature_eng(train)

test = feature_eng(test)
corr_matrix = train.corr()

corr_matrix.SalePrice.sort_values(ascending = False)
with sns.axes_style(style = "ticks"):

    sns.clustermap(corr_matrix, center=0, cmap="vlag", 

                   linewidths=.75, figsize=(13, 13))
train[["SalePrice", "OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF", "1stFlrSF", 

       "FullBath", "TotRmsAbvGrd", "YearBuilt", "YearRemodAdd", "GarageYrBlt", "MasVnrArea", "HouseAge",

      "TotalBathrooms", "TotalPorchSF"]].describe().T
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.base import BaseEstimator, TransformerMixin



from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import OneHotEncoder

unimportant_columns = ["Id", "Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"]

train.drop(columns=unimportant_columns, axis=1, inplace=True)

test.drop(columns=unimportant_columns, axis=1, inplace=True)
features, target = train.drop("SalePrice", axis = 1), train.SalePrice.to_numpy()
cats = [col for col in features.columns if features[col].dtype == object]

nums = [col for col in features.columns if col not in cats]
features[nums] = features[nums].replace([np.inf, -np.inf], np.nan)

features[nums] = features[nums].astype('float')



test[nums] = test[nums].replace([np.inf, -np.inf], np.nan)

test[nums] = test[nums].astype('float')
class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

        

    def fit(self, X, y = None):

        return self

    

    def transform(self, X):

        return X[self.attribute_names].values
def full_preparation(cats, nums):

    num_pipeline = Pipeline([    

        ("selector", DataFrameSelector(nums)),

        ("imputer", SimpleImputer(strategy="median")),

        ("std_scaler", MinMaxScaler(feature_range = (0, 1))),

    ])

    cat_pipeline = Pipeline([    

        ("selector", DataFrameSelector(cats)),

        ("imputer1", SimpleImputer(strategy="most_frequent", missing_values = np.nan)),

        ("cat_encoder", OneHotEncoder(sparse = False, drop = "first")),

    ])

    full_pipeline = FeatureUnion(transformer_list = [

        ("num_pipeline", num_pipeline),

        ("cat_pipeline", cat_pipeline),

    ])

    

    return full_pipeline
prep = full_preparation(cats, nums).fit(features)
features_prep = prep.transform(features)

test_prep = prep.transform(test)
# Evaluate the models

from catboost import CatBoostRegressor

from sklearn.metrics import accuracy_score, mean_absolute_error
model = CatBoostRegressor(thread_count=4)
%%time



grid = {

    'learning_rate': [0.03, 0.1],

    'depth': [4, 6, 10],

    'l2_leaf_reg': [1, 3, 5, 7, 9],

    "border_count":[4, 10],

}



grid_search_result = model.grid_search(

    param_grid=grid,

    X=features_prep,

    y=target,

    train_size=0.8, #valid_size=0.2

    cv=3,

    plot=True,

)
grid_search_result["params"]
model.best_score_
predictions = model.predict(test_prep).tolist()
submission.head()
submission.SalePrice = predictions

submission.to_csv("submission.csv", index = False)