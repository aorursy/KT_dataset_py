import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib nbagg

import xgboost as xgb

import seaborn as sns
train = pd.read_csv("../input/train.csv", encoding= "utf_8")

test = pd.read_csv("../input/test.csv", encoding= "utf_8")
first_feat = ["id","timestamp","price_doc", "full_sq", "life_sq",

"floor", "max_floor", "material", "build_year", "num_room",

"kitch_sq", "state", "product_type", "sub_area"]
first_feat = ["id","timestamp", "full_sq", "life_sq",

"floor", "max_floor", "material", "build_year", "num_room",

"kitch_sq", "state", "product_type", "sub_area"]
bad_index = train[train.life_sq > train.full_sq].index

train.ix[bad_index, "life_sq"] = np.NaN
equal_index = [601,1896,2791]

test.ix[equal_index, "life_sq"] = test.ix[equal_index, "full_sq"]
bad_index = test[test.life_sq > test.full_sq].index

test.ix[bad_index, "life_sq"] = np.NaN
bad_index = train[train.life_sq < 5].index

train.ix[bad_index, "life_sq"] = np.NaN
bad_index = test[test.life_sq < 5].index

test.ix[bad_index, "life_sq"] = np.NaN
bad_index = train[train.full_sq < 5].index

train.ix[bad_index, "full_sq"] = np.NaN
bad_index = test[test.full_sq < 5].index

test.ix[bad_index, "full_sq"] = np.NaN
kitch_is_build_year = [13117]

train.ix[kitch_is_build_year, "build_year"] = train.ix[kitch_is_build_year, "kitch_sq"]
bad_index = train[train.kitch_sq >= train.life_sq].index

train.ix[bad_index, "kitch_sq"] = np.NaN
bad_index = test[test.kitch_sq >= test.life_sq].index

test.ix[bad_index, "kitch_sq"] = np.NaN
bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index

train.ix[bad_index, "kitch_sq"] = np.NaN
bad_index = test[(test.kitch_sq == 0).values + (test.kitch_sq == 1).values].index

test.ix[bad_index, "kitch_sq"] = np.NaN
bad_index = train[(train.full_sq > 210) * (train.life_sq / train.full_sq < 0.3)].index

train.ix[bad_index, "full_sq"] = np.NaN
bad_index = test[(test.full_sq > 150) * (test.life_sq / test.full_sq < 0.3)].index

test.ix[bad_index, "full_sq"] = np.NaN
bad_index = train[train.life_sq > 300].index

train.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN
bad_index = test[test.life_sq > 200].index

test.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN
train.product_type.value_counts(normalize= True)
test.product_type.value_counts(normalize= True)
bad_index = train[train.build_year < 1500].index

train.ix[bad_index, "build_year"] = np.NaN
bad_index = test[test.build_year < 1500].index

test.ix[bad_index, "build_year"] = np.NaN
bad_index = train[train.num_room == 0].index 

train.ix[bad_index, "num_room"] = np.NaN
bad_index = test[test.num_room == 0].index 

test.ix[bad_index, "num_room"] = np.NaN
bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]

train.ix[bad_index, "num_room"] = np.NaN
bad_index = [3174, 7313]

test.ix[bad_index, "num_room"] = np.NaN
bad_index = train[(train.floor == 0).values * (train.max_floor == 0).values].index

train.ix[bad_index, ["max_floor", "floor"]] = np.NaN
bad_index = train[train.floor == 0].index

train.ix[bad_index, "floor"] = np.NaN
bad_index = train[train.max_floor == 0].index

train.ix[bad_index, "max_floor"] = np.NaN
bad_index = test[test.max_floor == 0].index

test.ix[bad_index, "max_floor"] = np.NaN
bad_index = train[train.floor > train.max_floor].index

train.ix[bad_index, "max_floor"] = np.NaN
bad_index = test[test.floor > test.max_floor].index

test.ix[bad_index, "max_floor"] = np.NaN
train.floor.describe(percentiles= [0.9999])
bad_index = [23584]

train.ix[bad_index, "floor"] = np.NaN
train.material.value_counts()
test.material.value_counts()
train.state.value_counts()
bad_index = train[train.state == 33].index

train.ix[bad_index, "state"] = np.NaN
test.state.value_counts()
test.to_csv("test_clean.csv", index= False, encoding= "utf_8")

train.to_csv("train_clean.csv", index = False, encoding= "utf_8")