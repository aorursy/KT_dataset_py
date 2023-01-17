import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.linear_model import LinearRegression, Lasso

%matplotlib inline

sns.set()
full_data_labeled = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

full_data_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

full_data_combined = pd.concat([full_data_labeled.drop("SalePrice", axis=1), full_data_submission], keys=["labeled", "submission"], axis=0)

feature_cols = full_data_combined.columns[1:]

feature_types = pd.Series(index=feature_cols, data=["Nominal","Nominal","Continuous","Continuous","Nominal","Nominal","Ordinal","Nominal","Nominal","Nominal","Nominal","Nominal","Nominal","Nominal","Nominal","Nominal","Ordinal","Ordinal","Discrete","Discrete","Nominal","Nominal","Nominal","Nominal","Nominal","Continuous","Ordinal","Ordinal","Nominal","Ordinal","Ordinal","Ordinal","Ordinal","Continuous","Ordinal","Continuous","Continuous","Continuous","Nominal","Ordinal","Binary","Nominal","Continuous","Continuous","Continuous","Continuous","Discrete","Discrete","Discrete","Discrete","Discrete","Discrete","Ordinal","Discrete","Ordinal","Discrete","Ordinal","Nominal","Discrete","Ordinal","Discrete","Continuous","Ordinal","Ordinal","Nominal","Continuous","Continuous","Continuous","Continuous","Continuous","Continuous","Ordinal","Nominal","Nominal","Continuous","Discrete","Discrete","Nominal","Nominal"])
correlation_matrix = full_data_labeled.corr()

figures, axes = plt.subplots(1,1, figsize=(18,18))

sns.heatmap(correlation_matrix, vmax=0.8, square=True)
def plot_feature(feature):

    if isinstance(feature, int):

        feature = feature_cols[feature]

    print("Feature: " + feature)

    if feature_types[feature] == "Continuous":

        figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 12))

        sns.distplot(a=full_data_labeled[feature].dropna(), ax=axes[0])

        sns.regplot(x=feature, y="SalePrice", data=full_data_labeled, ax=axes[1])

    if feature_types[feature] == "Discrete" or feature_types[feature] == "Nominal" or feature_types[feature] == "Binary" or feature_types[feature] == "Ordinal":

        figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 12))

        sns.countplot(x=full_data_labeled[feature].dropna(), ax=axes[0])

        sns.boxplot(x=feature, y="SalePrice", data=full_data_labeled, ax=axes[1])
plot_feature(0)
most_correlated_features = correlation_matrix.nlargest(9, "SalePrice")["SalePrice"].index[1:] # 8 most correlated values with SalePrice

most_correlated_features
plot_feature("OverallQual")
plot_feature("GrLivArea")
plot_feature("GarageCars")
plot_feature("GarageArea")
plot_feature("TotalBsmtSF")
plot_feature("1stFlrSF")
plot_feature("FullBath")
plot_feature("TotRmsAbvGrd")
num_missing_data = full_data_combined.isnull().sum()

num_missing_data = num_missing_data[num_missing_data > 0].sort_values(ascending=False)

with pd.option_context('display.max_rows', None):

    print(num_missing_data)
full_data_combined_categorical_filled = full_data_combined.copy()

NA_cols = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature", "MasVnrType", "MSZoning", "Functional", "Utilities", "Exterior1st", "Exterior2nd", "SaleType", "Electrical", "KitchenQual"]

for col in NA_cols:

    full_data_combined_categorical_filled[col] = full_data_combined_categorical_filled[col].replace(np.nan, "LA") # LA for "lacking"

    

num_missing_data = full_data_combined_categorical_filled.isnull().sum()

num_missing_data = num_missing_data[num_missing_data > 0].sort_values(ascending=False)

with pd.option_context('display.max_rows', None):

    print(num_missing_data)
full_data_combined.loc[full_data_combined["LotFrontage"].isnull(), ["LotArea", "LotShape", "LotConfig"]]
figrues, axes = plt.subplots(1,3,figsize=(36,12))

sns.regplot(x="LotArea", y="LotFrontage", data=full_data_combined, ax=axes[0])

sns.boxplot(x="LotConfig", y="LotFrontage", data=full_data_combined, ax=axes[1])

sns.boxplot(x="LotShape", y="LotFrontage", data=full_data_combined, ax=axes[2])

sns.relplot(x="LotArea", y="LotFrontage", hue="LotConfig", size="LotShape", data=full_data_combined)
lot_dummies = pd.get_dummies(full_data_combined_categorical_filled[["LotFrontage", "LotArea", "LotShape", "LotConfig"]])

lot_features = lot_dummies.dropna()

lot_missing = lot_dummies[:][lot_dummies["LotFrontage"].isnull()]



# manually remove an outlier

outlier_index = lot_features[lot_features["LotArea"] > 200000].index

lot_features = lot_features.drop(outlier_index, axis=0)



imputer = LinearRegression()

imputer.fit(lot_features.drop("LotFrontage", axis=1), lot_features["LotFrontage"])

lot_missing["LotFrontage"] = imputer.predict(lot_missing.drop("LotFrontage", axis=1))

full_data_combined_categorical_lot_filled = full_data_combined_categorical_filled.copy()

full_data_combined_categorical_lot_filled["LotFrontage"] = full_data_combined_categorical_filled["LotFrontage"].combine_first(lot_missing["LotFrontage"])
full_data_combined_categorical_lot_filled.loc[full_data_combined_categorical_lot_filled["GarageYrBlt"].isnull(), "GarageArea"]
full_data_combined_filled = full_data_combined_categorical_lot_filled.fillna(full_data_combined_categorical_lot_filled.mean())

full_data_combined_filled.isnull().sum().sum()
numeric_features_discrete = full_data_combined_filled.drop("Id", axis=1).loc[:, feature_types == "Discrete"]

numeric_features_continuous = full_data_combined_filled.drop("Id", axis=1).loc[:, feature_types == "Continuous"]

numeric_features = pd.concat([numeric_features_discrete, numeric_features_continuous], axis=1)



figures, axes = plt.subplots(9, 4, figsize=(18, 70))

for i in range(9):

    for j in range(4):

        if i*4+j >= numeric_features.shape[1]:

            continue

        sns.distplot(a=numeric_features.iloc[:, i*4+j], ax=axes[i, j])
scaler = MinMaxScaler()

numeric_features_transformed = pd.DataFrame(scaler.fit_transform(numeric_features))



figures, axes = plt.subplots(9, 4, figsize=(18, 70))

for i in range(9):

    for j in range(4):

        if i*4+j >= numeric_features_transformed.shape[1]:

            continue

        sns.distplot(a=numeric_features_transformed.iloc[:, i*4+j], ax=axes[i, j])



numeric_features_labeled = numeric_features_transformed.iloc[:1460]

numeric_features_submission = numeric_features_transformed.iloc[1460:]
categorical_features_nominal = full_data_combined_filled.drop("Id", axis=1).loc[:, feature_types == "Nominal"]

categorical_features_ordinal = full_data_combined_filled.drop("Id", axis=1).loc[:, feature_types == "Ordinal"]

categorical_features_binary = full_data_combined_filled.drop("Id", axis=1).loc[:, feature_types == "Binary"]

categorical_features = pd.concat([categorical_features_nominal, categorical_features_ordinal, categorical_features_binary], axis=1)

categorical_features = pd.get_dummies(categorical_features)

categorical_features_labeled = categorical_features.xs("labeled")

categorical_features_submission = categorical_features.xs("submission")

categorical_features_submission["index"] = range(1460, 2919)

categorical_features_submission.set_index("index", inplace=True)
data_labeled = pd.concat([numeric_features_labeled, categorical_features_labeled], axis=1)

data_submission = pd.concat([numeric_features_submission, categorical_features_submission], axis=1)

print(data_labeled.shape)

print(data_submission.shape)