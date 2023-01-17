from pathlib import Path



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



import missingno as msno



sns.set_style('whitegrid')

%matplotlib inline



training_dataset_path = Path("../input/train.csv")

testing_dataset_path= Path("../input/test.csv")



train = pd.read_csv(training_dataset_path)

test = pd.read_csv(testing_dataset_path)

print(f"Training dataset shape: {train.shape}")

print(f"Testing dataset shape: {test.shape}")
categorical_features = train.select_dtypes(include=[np.object]).columns.values

numerical_features = train.select_dtypes(include=[np.number]).columns.values



feature_names = {"Categorical": categorical_features, "Numerical": numerical_features}

# For readability sake, I transpose dataframe so categories are columns and features are rows

feature_types = pd.DataFrame.from_dict(feature_names, orient="index").T



# Fillna so last rows do not contain nans

display(feature_types.fillna(''))

display(feature_types.count())
ordinal_features = np.array([

    "ExterQual",

    "ExterCond",

    "BsmtQual",

    "BsmtCond",

    "BsmtExposure",

    "HeatingQC",

    "KitchenQual",

    "Functional",

    "FireplaceQu",

    "GarageFinish",

    "GarageQual",

    "GarageCond",

    "PavedDrive",

    "PoolQC",

    "Utilities",

    "BsmtFinType1",

    "BsmtFinType2",

    "LandSlope",

    "Electrical",

    "Fence"

])
nominal_features = np.setdiff1d(categorical_features, ordinal_features)
feature_names = {

    "Ordinal": ordinal_features,

    "Nominal": nominal_features,

    "Numerical": numerical_features,

}



feature_types = pd.DataFrame.from_dict(feature_names, orient="index").T

display(feature_types.fillna(''))

display(feature_types.count())
full = pd.concat([train, test], keys=['train', 'test'], sort=False)

display(full.head())

display(full.tail())
def nan_rows(dataset: pd.DataFrame):

    nan_rows_count = dataset.isnull().any(axis=1).sum()

    return nan_rows_count, nan_rows_count / len(dataset) * 100





def nan_features(dataset: pd.DataFrame):

    nans_per_feature = dataset.isnull().sum().sort_values(ascending=False)

    nan_features = nans_per_feature[nans_per_feature != 0].reset_index()

    nan_features.columns = ["Feature", "NaNs"]

    return nan_features





def nan_count(dataset: pd.DataFrame):

    return dataset.isnull().sum().sum()





def display_nan_statistics(

    dataset: pd.DataFrame, remove_target: bool = True, target_name: str = "SalePrice"

):

    if remove_target:

        df = dataset.drop(target_name, axis=1)

    else:

        df = dataset

    print("Dataset contains {} NaNs".format(nan_count(df)))

    print("NaN rows: {} | In percentage: {}".format(*nan_rows(df)))

    print("NaNs per feature:")

    display(nan_features(df))





display_nan_statistics(full)
described_features = [

  "PoolQC",

  "MiscFeature",

  "Alley",

  "Fence",

  "FireplaceQu",

  "GarageCond",

  "GarageQual",

  "GarageFinish",

  "GarageType",

  "BsmtCond",

  "BsmtExposure",

  "BsmtQual",

  "BsmtFinType2",

  "BsmtFinType1",

]



full.fillna(value={feature: "ValueAbsent" for feature in described_features}, inplace=True)



display_nan_statistics(full)
full = full[full["Electrical"].notnull()]
garage_data = full.filter(regex=".*Garage.*")

garage_data = garage_data.replace(["ValueAbsent", 0], np.NaN)

msno.heatmap(garage_data)
def falsely_described_nans(dataset):

  # If any feature in the dataset is NaN while other is not it will be returned.

  indices = []

  features = dataset.columns.tolist()

  for index, nan_feature in enumerate(features):

    for non_nan_feature in features[index:]:

      df = dataset[

                dataset[nan_feature].isnull() &

                dataset[non_nan_feature].notnull()

              ]

      if not df.empty:

        indices.extend(tuple(df.index.tolist()))



  return dataset.loc[list(set(indices))].copy()
false_nans = falsely_described_nans(garage_data)

display(false_nans)
# Change GarageYrBlt values to zero

full.fillna(value={"GarageYrBlt": 0}, inplace=True)

# Get index and replace our imputed ValueAbsent with NaN 

full.at[false_nans.index, ["GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond"]] = np.NaN



display_nan_statistics(full)
basement_data = full.filter(regex=".*Bsmt.*").copy()

basement_data.replace(["ValueAbsent"], np.NaN, inplace=True)

msno.heatmap(basement_data)
described_features = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]

basement_described_data = basement_data[described_features]



false_nans = falsely_described_nans(basement_described_data)

display(false_nans)



display(msno.heatmap(basement_described_data))
full.at[

    false_nans.index,

    ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"],

] = np.NaN



display_nan_statistics(full)
year_features=[ "YearBuilt", "YearRemodAdd", "GarageYrBlt" ]

full[year_features].describe()
fake_years = full[full["GarageYrBlt"] > 2010]

display(fake_years)
full.at[fake_years.index, 'GarageYrBlt'] = 2007
fake_remodel = full[full["YearBuilt"] > full["YearRemodAdd"]][["YearBuilt", "YearRemodAdd"]]

fake_remodel
full.at[fake_remodel.index, 'YearRemodAdd'] = 2002
# Go through all the possible contradictions to description:

def contradictive_area_quality(

    dataset,

    area_feature: str,

    quality_feature: str,

    absent_value_string: str = "ValueAbsent",

):

    area_no_quality = full[

        (dataset[area_feature] != 0) & (full[quality_feature] == absent_value_string)

    ].copy()



    quality_no_area = full[

        (dataset[area_feature] == 0) & (full[quality_feature] != absent_value_string)

    ].copy()



    return pd.concat([area_no_quality, quality_no_area])



print("Contradictive pools:")

pools = contradictive_area_quality(full, "PoolArea", "PoolQC")

display(pools[["PoolArea", "PoolQC"]])



print("Contradictive masonry veneer:")

veneers = contradictive_area_quality(full, "MasVnrArea", "MasVnrType", "None")

display(veneers[["MasVnrArea", "MasVnrType"]])



print("Lot Area false variables")

zero_lot_data = full[full["LotArea"] == 0].copy()

display(zero_lot_data)
# Pool Area looks legit, I suppose poll quality is simply missing

full.at[pools.index, "PoolQC"] = np.NaN



# 1.0 veneer area is ridiculously small, maybe a simple mistakes, change to 0

indices = pd.MultiIndex.from_tuples([("train", 773), ("train", 1230), ("test", 992)])

full.at[indices, "MasVnrArea"] = 0



# MasVnrType does not seem random, input NaN and leave it for imputation

indices = pd.MultiIndex.from_tuples([("train", 688), ("train", 1241), ("test", 859)])

full.at[indices, "MasVnrArea"] = np.NaN



# MasVnrArea looks legit, the type seems to be simply missing

indices = pd.MultiIndex.from_tuples(

    [("train", 624), ("train", 1300), ("train", 1334), ("test", 209)]

)

full.at[indices, "MasVnrType"] = np.NaN
ordinal_mapping = {

  "Ex": 2,

  "Gd": 1,

  "TA": 0,

  "Fa": -1,

  "Po": -2,

  "ValueAbsent": -3, # our designed NaN placeholder



  "Gd":	1,

  "Av":	0,

  "Mn":	-1,

  "No":	-2,



  "GLQ": 3,

  "ALQ": 2,

  "BLQ": 1,

  "Rec": 0,

  "LwQ": -1,

  "Unf": -2,



  "Typ" : 3,

  "Min1": 2,

  "Min2": 1,

  "Mod" : 0,

  "Maj1": -1,

  "Maj2": -2,

  "Sev" : -3,

  "Sal" : -4,



  "Fin": 0,

  "RFn": -1,

  "Unf": -2,



  "Y": 1,

  "P": 0,

  "N": -1,



  "AllPub": 1,

  "NoSewr": 0,

  "NoSeWa": -1,

  "ELO": -2,



  "Gtl": 1,

  "Severe": -1

}



full.replace({"LandSlope": "Sev"}, "Severe")

full = full.replace(ordinal_mapping)



display_nan_statistics(full)
initially_preprocessed_train = "../input/initially_preprocessed_train.csv"

initially_preprocessed_test = "../input/initially_preprocessed_test.csv"



train = full.loc["train", :]

test = full.loc["test", :]



# Can't save here, you might do it locally though :)

# train.to_csv(initially_preprocessed_train)

# test.to_csv(initially_preprocessed_test)