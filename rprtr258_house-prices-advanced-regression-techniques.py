import matplotlib.pyplot as plt

import seaborn as sns; sns.set_style("whitegrid")



import numpy as np



import pandas as pd



from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline



from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.compose import TransformedTargetRegressor



from sklearn.metrics import make_scorer

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
iowa_file_path = '/kaggle/input/house-prices-advanced-regression-techniques/train.csv'

test_file_path = '/kaggle/input/house-prices-advanced-regression-techniques/test.csv'



home_data = pd.read_csv(iowa_file_path, index_col="Id")
home_data.shape
for feature_name, dtype in zip(home_data.columns, home_data.dtypes):

    print(f"{feature_name:12s}: {dtype}")
print("Before applying log")

print(f"SalePrice skew: {home_data['SalePrice'].skew():.3f}")

sns.distplot(home_data['SalePrice'], hist=False)

plt.show()

print("After applying log")

logy = np.log(home_data['SalePrice'])

print(f"SalePrice skew: {logy.skew():.3f}")

sns.distplot(logy, hist=False)

plt.show()
data = home_data.corr()["SalePrice"].sort_values()[::-1]

plt.figure(figsize=(12, 8))

sns.barplot(x=data.values, y=data.index)

plt.title("Correlation with SalePrice")

plt.xlim(-0.2, 1)

plt.show()
data = home_data.pivot_table(index="OverallQual", values="SalePrice", aggfunc="mean")

sns.barplot(x=data.index, y=data["SalePrice"])

plt.show()
data = home_data[["GrLivArea", "SalePrice"]]

sns.regplot(x=np.log(data["GrLivArea"]), y=np.log(data["SalePrice"]))

plt.show()
sns.regplot(x="GarageArea", y="SalePrice", data=home_data[["GarageArea", "SalePrice"]])

plt.xlim(0, 1500)

plt.ylim(0, 800000)

plt.show()

data = home_data[["GarageArea", "SalePrice"]]

data = data[data["GarageArea"] > 0]

sns.regplot(x="GarageArea", y="SalePrice", data=data)

plt.xlim(0, 1500)

plt.ylim(0, 800000)

plt.show()
data_tmp = pd.DataFrame(columns=["GarageArea", "SalePrice"])

for garage_area, sale_price in home_data[["GarageArea", "SalePrice"]].values:

    data_tmp = data_tmp.append(pd.DataFrame([[">0" if garage_area > 0 else "=0", sale_price]], columns=data_tmp.columns))

data_tmp["SalePrice"] = data_tmp["SalePrice"].apply(float)

sns.boxplot(x="GarageArea", y="SalePrice", data=data_tmp)

plt.show()
data = home_data[["GarageCars", "SalePrice"]]

sns.boxplot(data["GarageCars"], y=data["SalePrice"])

plt.show()

data = home_data["GarageCars"].value_counts()

print("GarageCars: Entries count")

print(data)

sns.barplot(x=data.index, y=data.values)

plt.show()
data = home_data[["MSZoning", "SalePrice"]]

sns.boxplot(x="MSZoning", y="SalePrice", data=data)

plt.show()
data = home_data[["Alley", "SalePrice"]]

sns.boxplot(x="Alley", y="SalePrice", data=data)

plt.show()
data = home_data[["LotShape", "SalePrice"]]

sns.boxplot(x="LotShape", y="SalePrice", data=data)

plt.show()
data = home_data[["LandContour", "SalePrice"]]

sns.boxplot(x="LandContour", y="SalePrice", data=data)

plt.show()
data = home_data[["Utilities", "SalePrice"]]

sns.boxplot(x="Utilities", y="SalePrice", data=data)

plt.show()
data = home_data[["LotConfig", "SalePrice"]]

sns.boxplot(x="LotConfig", y="SalePrice", data=data)

plt.show()
data = home_data[["LandSlope", "SalePrice"]]

sns.boxplot(x="LandSlope", y="SalePrice", data=data)

plt.show()
## Neighborhood

data = home_data[["Neighborhood", "SalePrice"]]

plt.figure(figsize=(10, 5))

sns.boxplot(x="Neighborhood", y="SalePrice", data=data)

plt.show()
## Condition1

data = home_data[["Condition1", "SalePrice"]]

sns.boxplot(x="Condition1", y="SalePrice", data=data)

plt.show()
## Condition2

data = home_data[["Condition2", "SalePrice"]]

sns.boxplot(x="Condition2", y="SalePrice", data=data)

plt.show()
## BldgType

data = home_data[["BldgType", "SalePrice"]]

sns.boxplot(x="BldgType", y="SalePrice", data=data)

plt.show()
## HouseStyle

data = home_data[["HouseStyle", "SalePrice"]]

sns.boxplot(x="HouseStyle", y="SalePrice", data=data)

plt.show()
## RoofStyle

data = home_data[["RoofStyle", "SalePrice"]]

sns.boxplot(x="RoofStyle", y="SalePrice", data=data)

plt.show()
## RoofMatl

data = home_data[["RoofMatl", "SalePrice"]]

sns.boxplot(x="RoofMatl", y="SalePrice", data=data)

plt.show()
## Exterior1st

data = home_data[["Exterior1st", "SalePrice"]]

sns.boxplot(x="Exterior1st", y="SalePrice", data=data)

plt.show()
## Exterior2nd

data = home_data[["Exterior2nd", "SalePrice"]]

sns.boxplot(x="Exterior2nd", y="SalePrice", data=data)

plt.show()
## MasVnrType

data = home_data[["MasVnrType", "SalePrice"]]

sns.boxplot(x="MasVnrType", y="SalePrice", data=data)

plt.show()
## ExterQual

data = home_data[["ExterQual", "SalePrice"]]

sns.boxplot(x="ExterQual", y="SalePrice", data=data)

plt.show()
## ExterCond

data = home_data[["ExterCond", "SalePrice"]]

sns.boxplot(x="ExterCond", y="SalePrice", data=data)

plt.show()
## Foundation

data = home_data[["Foundation", "SalePrice"]]

sns.boxplot(x="Foundation", y="SalePrice", data=data)

plt.show()
## BsmtQual

data = home_data[["BsmtQual", "SalePrice"]]

sns.boxplot(x="BsmtQual", y="SalePrice", data=data)

plt.show()
## BsmtCond

data = home_data[["BsmtCond", "SalePrice"]]

sns.boxplot(x="BsmtCond", y="SalePrice", data=data)

plt.show()
## BsmtExposure

data = home_data[["BsmtExposure", "SalePrice"]]

sns.boxplot(x="BsmtExposure", y="SalePrice", data=data)

plt.show()
## BsmtFinType1

data = home_data[["BsmtFinType1", "SalePrice"]]

sns.boxplot(x="BsmtFinType1", y="SalePrice", data=data)

plt.show()
## BsmtFinType2

data = home_data[["BsmtFinType2", "SalePrice"]]

sns.boxplot(x="BsmtFinType2", y="SalePrice", data=data)

plt.show()
## Heating

data = home_data[["Heating", "SalePrice"]]

sns.boxplot(x="Heating", y="SalePrice", data=data)

plt.show()
## HeatingQC

data = home_data[["HeatingQC", "SalePrice"]]

sns.boxplot(x="HeatingQC", y="SalePrice", data=data)

plt.show()
## CentralAir

data = home_data[["CentralAir", "SalePrice"]]

sns.boxplot(x="CentralAir", y="SalePrice", data=data)

plt.show()
## Electrical

data = home_data[["Electrical", "SalePrice"]]

sns.boxplot(x="Electrical", y="SalePrice", data=data)

plt.show()
## KitchenQual

data = home_data[["KitchenQual", "SalePrice"]]

sns.boxplot(x="KitchenQual", y="SalePrice", data=data)

plt.show()
## Functional

data = home_data[["Functional", "SalePrice"]]

sns.boxplot(x="Functional", y="SalePrice", data=data)

plt.show()
## FireplaceQu

data = home_data[["FireplaceQu", "SalePrice"]]

sns.boxplot(x="FireplaceQu", y="SalePrice", data=data)

plt.show()
## GarageType

data = home_data[["GarageType", "SalePrice"]]

sns.boxplot(x="GarageType", y="SalePrice", data=data)

plt.show()
## GarageFinish

data = home_data[["GarageFinish", "SalePrice"]]

sns.boxplot(x="GarageFinish", y="SalePrice", data=data)

plt.show()
## GarageQual

data = home_data[["GarageQual", "SalePrice"]]

sns.boxplot(x="GarageQual", y="SalePrice", data=data)

plt.show()
## GarageCond

data = home_data[["GarageCond", "SalePrice"]]

sns.boxplot(x="GarageCond", y="SalePrice", data=data)

plt.show()
## PavedDrive

data = home_data[["PavedDrive", "SalePrice"]]

sns.boxplot(x="PavedDrive", y="SalePrice", data=data)

plt.show()
## PoolQC

data = home_data[["PoolQC", "SalePrice"]]

sns.boxplot(x="PoolQC", y="SalePrice", data=data)

plt.show()
## Fence

data = home_data[["Fence", "SalePrice"]]

sns.boxplot(x="Fence", y="SalePrice", data=data)

plt.show()
## MiscFeature

data = home_data[["MiscFeature", "SalePrice"]]

sns.boxplot(x="MiscFeature", y="SalePrice", data=data)

plt.show()
## SaleType

data = home_data[["SaleType", "SalePrice"]]

sns.boxplot(x="SaleType", y="SalePrice", data=data)

plt.show()
## SaleCondition

data = home_data[["SaleCondition", "SalePrice"]]

sns.boxplot(x="SaleCondition", y="SalePrice", data=data)

plt.show()
nulls = pd.DataFrame(home_data.isnull().sum().sort_values(ascending=False))

nulls.columns = ["Null count"]

nulls["Unique"] = home_data[nulls.index].nunique()

nulls["dtype"] = home_data[nulls.index].dtypes

nulls[nulls["Null count"] > 0]
home_data["GarageArea=0"] = (home_data["GarageArea"] == 0).apply(float)

home_data["GarageCars=4"] = (home_data["GarageCars"] == 4).apply(float)

home_data["MSSubClass"] = home_data["MSSubClass"].apply(str)



home_data["NotHasBasement"] = home_data["BsmtExposure"].isnull()

home_data["BsmtExposure"] = home_data["BsmtExposure"].fillna("NoBasement")

home_data["BsmtQual"] = home_data["BsmtQual"].fillna("NoBasement")

home_data["BsmtCond"] = home_data["BsmtCond"].fillna("NoBasement")

home_data["BsmtFinType1"] = home_data["BsmtFinType1"].fillna("NoBasement")

home_data["BsmtFinType2"] = home_data["BsmtFinType2"].fillna("NoBasement")



home_data["NotHasFireplace"] = home_data["FireplaceQu"].isnull()

home_data["FireplaceQu"] = home_data["FireplaceQu"].fillna("NoFireplace")



home_data["NotHasGarage"] = home_data["GarageFinish"].isnull()

home_data["GarageFinish"] = home_data["GarageFinish"].fillna("NoGarage")

home_data["GarageQual"] = home_data["GarageQual"].fillna("NoGarage")

home_data["GarageType"] = home_data["GarageType"].fillna("NoGarage")

home_data["GarageCond"] = home_data["GarageCond"].fillna("NoGarage")



home_data["NotHasPool"] = home_data["PoolQC"].isnull()

home_data["PoolQC"] = home_data["PoolQC"].fillna("NoPool")



home_data["NotHasFence"] = home_data["Fence"].isnull()

home_data["Fence"] = home_data["Fence"].fillna("NoFence")



home_data["NotHasKitchen"] = home_data["KitchenQual"].isnull()

home_data["KitchenQual"] = home_data["KitchenQual"].fillna("NoKitchen")



#home_data["MiscFeature"] = home_data["MiscFeature"].fillna("NA")

#home_data["Alley"] = home_data["Alley"].fillna("NA")

#home_data["LotFrontage"] = home_data["LotFrontage"].fillna(home_data["LotFrontage"].min())

#home_data["GarageYrBlt"] = home_data["GarageYrBlt"].fillna(home_data["GarageYrBlt"].min())

#home_data["MasVnrType"] = home_data["MasVnrType"].fillna("NA")

#home_data["MasVnrArea"] = home_data["MasVnrArea"].fillna(home_data["MasVnrArea"].min())

#home_data["Electrical"] = home_data["Electrical"].fillna("NA")
def show_nulls(df):

    nulls = pd.DataFrame(df.isnull().sum().sort_values(ascending=False))

    nulls.columns = ["Null count"]

    nulls["Unique"] = df[nulls.index].nunique()

    nulls["dtype"] = df[nulls.index].dtypes

    return nulls[nulls["Null count"] > 0]

show_nulls(home_data)
y = home_data["SalePrice"].values

X = home_data.drop("SalePrice", axis=1)
models_performance = pd.DataFrame(columns=["Model", "CV", "Full"])



def calc_score(y_true, y_pred):

    logy_true = np.log1p(y_true)

    logy_pred = np.log1p(y_pred)

    return np.sqrt(np.sum((logy_true - logy_pred) ** 2) / y_true.shape[0]) 



def test_pipeline(pipeline, name="Unnamed"):

    # Uncomment when commit

    return

    scores = cross_val_score(pipeline, X, y, cv=10, scoring=make_scorer(calc_score))

    print(f"Mean CV score: {scores.mean():,.5f}")



    pipeline.fit(X, y)

    full_score = calc_score(y, pipeline.predict(X))

    print(f"Full score: {full_score:,.5f}")

    

    global models_performance

    models_performance = models_performance.append(pd.DataFrame([[name, scores.mean(), full_score]], columns=models_performance.columns))
def log_model(model):

    return TransformedTargetRegressor(model, func=np.log1p, inverse_func=np.expm1)
numerical_cols = X.select_dtypes(exclude=['object']).columns

ordinal_cols = ["LotShape", "LandContour", "LandSlope", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "HeatingQC", "CentralAir", "KitchenQual", "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence"]

categorical_cols = [col for col in X.select_dtypes(include=["object"]).columns if col not in ordinal_cols]



# Impute numerical features with mean

# Impute categoricals and onehot encode them

preprocessor = ColumnTransformer(transformers=[

    ('num', SimpleImputer(strategy='mean'), numerical_cols),

    ('cat', Pipeline(steps=[

        ('imputer', SimpleImputer(strategy='constant')),

        ('onehot', OneHotEncoder(handle_unknown='ignore'))

    ]), categorical_cols)

##     ('num', SimpleImputer(strategy='mean', verbose=2), numerical_cols),

##     ('imp', SimpleImputer(strategy='constant', verbose=2), ordinal_cols),

##     ('const', SimpleImputer(strategy='constant', verbose=2), categorical_cols),

##     ('onehot encode', OneHotEncoder(handle_unknown='ignore'), categorical_cols)

#     ('encode Street', OrdinalEncoder(categories=[["Grvl", "Pave"]]), ["Street"]),

#     ('encode LotShape', OrdinalEncoder(categories=[["Reg", "IR1", "IR2", "IR3"]]), ["LotShape"]),

#     ('encode LandContour', OrdinalEncoder(categories=[["Lvl", "Bnk", "HLS", "Low"]]), ["LandContour"]),

#     ('encode LandSlope', OrdinalEncoder(categories=[["Gtl", "Mod", "Sev"]]), ["LandSlope"]),

#     ('encode ExterQual', OrdinalEncoder(categories=[["Ex", "Gd", "TA", "Fa", "Po"]]), ["ExterQual"]),

#     ('encode ExterCond', OrdinalEncoder(categories=[["Ex", "Gd", "TA", "Fa", "Po"]]), ["ExterCond"]),

#     ('encode BsmtQual', OrdinalEncoder(categories=[["Ex", "Gd", "TA", "Fa", "Po", "NoBasement"]]), ["BsmtQual"]),

#     ('encode BsmtCond', OrdinalEncoder(categories=[["Ex", "Gd", "TA", "Fa", "Po", "NoBasement"]]), ["BsmtCond"]),

#     ('encode BsmtExposure', OrdinalEncoder(categories=[["Gd", "Av", "Mn", "No", "NoBasement"]]), ["BsmtExposure"]),

#     ('encode BsmtFinType1', OrdinalEncoder(categories=[["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NoBasement"]]), ["BsmtFinType1"]),

#     ('encode BsmtFinType2', OrdinalEncoder(categories=[["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NoBasement"]]), ["BsmtFinType2"]),

#     ('encode HeatingQC', OrdinalEncoder(categories=[["Ex", "Gd", "TA", "Fa", "Po"]]), ["HeatingQC"]),

#     ('encode CentralAir', OrdinalEncoder(categories=[["Y", "N"]]), ["CentralAir"]),

#     ('encode KitchenQual', OrdinalEncoder(categories=[["Ex", "Gd", "TA", "Fa", "Po"]]), ["KitchenQual"]),

#     ('encode FireplaceQu', OrdinalEncoder(categories=[["Ex", "Gd", "TA", "Fa", "Po", "NoFireplace"]]), ["FireplaceQu"]),

#     ('encode GarageFinish', OrdinalEncoder(categories=[["Fin", "RFn", "Unf", "NoGarage"]]), ["GarageFinish"]),

#     ('encode GarageQual', OrdinalEncoder(categories=[["Ex", "Gd", "TA", "Fa", "Po", "NoGarage"]]), ["GarageQual"]),

#     ('encode GarageCond', OrdinalEncoder(categories=[["Ex", "Gd", "TA", "Fa", "Po", "NoGarage"]]), ["GarageCond"]),

#     ('encode PavedDrive', OrdinalEncoder(categories=[["Y", "P", "N"]]), ["PavedDrive"]),

#     ('encode PoolQC', OrdinalEncoder(categories=[["Ex", "Gd", "TA", "Fa", "Po", "NoPool"]]), ["PoolQC"]),

#     ('encode Fence', OrdinalEncoder(categories=[["GdPrv", "MnPrv", "GdWo", "MnWw", "NoFence"]]), ["Fence"])

])
lin2_pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('model', log_model(LinearRegression()))

])



test_pipeline(lin2_pipeline, "Linear regression[1]")
xg_pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('model', log_model(XGBRegressor(n_estimators=500, objective='reg:squarederror')))

])



test_pipeline(xg_pipeline, "XGBoost[1]")
from sklearn.ensemble import GradientBoostingRegressor



gbr_pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('model', log_model(GradientBoostingRegressor()))

])



test_pipeline(gbr_pipeline, "GradientBoostingRegressor[1]")
random_fores_pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('model', log_model(RandomForestRegressor(n_estimators=100, random_state=1)))

])



test_pipeline(random_fores_pipeline, "Random forest[1]")
from lightgbm import LGBMRegressor



lightgbm_pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('model', log_model(LGBMRegressor()))

])



test_pipeline(lightgbm_pipeline, "LightGBM[1]")
from sklearn.base import BaseEstimator



class Stacked(BaseEstimator):

    def __init__(self, models, stacker):

        self.models = models

        self.stacker = stacker

    def _meta_predict(self, X):

        metaX = []

        for model in self.models:

            fX = model.predict(X)

            metaX.append(fX)

        return np.vstack(metaX).T

    def fit(self, X, y):

        for model in self.models:

            model.fit(X, y)

        finX = self._meta_predict(X)

        self.stacker.fit(finX, y)

    def predict(self, X):

        finX = self._meta_predict(X)

        return np.abs(self.stacker.predict(finX))



stacked_model = Stacked([xg_pipeline, lightgbm_pipeline], LinearRegression())

test_pipeline(stacked_model, "LinearRegression(XGBoost, LightGBM)")
from sklearn.neural_network import MLPRegressor



class Stacked(BaseEstimator):

    def __init__(self, models, stacker):

        self.models = models

        self.stacker = stacker

    def _meta_predict(self, X):

        metaX = []

        for model in self.models:

            fX = model.predict(X)

            metaX.append(fX)

        return np.vstack(metaX).T

    def fit(self, X, y):

        for model in self.models:

            model.fit(X, y)

        finX = self._meta_predict(X)

        self.stacker.fit(finX, y)

    def predict(self, X):

        finX = self._meta_predict(X)

        return np.abs(self.stacker.predict(finX))



stacked_model2020 = Stacked([xg_pipeline, lightgbm_pipeline], MLPRegressor(learning_rate_init=0.01))

test_pipeline(stacked_model2020, "NN(XGBoost, LightGBM)")
class Stacked2(BaseEstimator):

    def __init__(self, models, stacker):

        self.models = models

        self.stacker = stacker

    def _meta_predict(self, X):

        metaX = [preprocessor.transform(X)]

        for model in self.models:

            fX = model.predict(X)

            metaX.append(fX.reshape(fX.shape[0], 1))

        return np.concatenate(metaX, axis=1)

    def fit(self, X, y):

        for model in self.models:

            model.fit(X, y)

        finX = self._meta_predict(X)

        self.stacker.fit(finX, y)

    def predict(self, X):

        finX = self._meta_predict(X)

        return np.abs(self.stacker.predict(finX))



stacked_model3 = Stacked2([xg_pipeline, lightgbm_pipeline], LinearRegression())

#test_pipeline(stacked_model3, "Stacked(XGBoost, LightGBM)")
stacked_model4 = Stacked([xg_pipeline, gbr_pipeline], LinearRegression())

test_pipeline(stacked_model4, "Stacked(XGBoost, GradientBoostingRegressor)")
models_performance.sort_values("CV")
test_X = pd.read_csv(test_file_path, index_col='Id')

test_X["GarageArea=0"] = (test_X["GarageArea"] == 0).apply(float)

test_X["GarageCars=4"] = (test_X["GarageCars"] == 4).apply(float)

test_X["MSSubClass"] = test_X["MSSubClass"].apply(str)



test_X["NotHasBasement"] = test_X["BsmtExposure"].isnull()

test_X["BsmtExposure"] = test_X["BsmtExposure"].fillna("NoBasement")

test_X["BsmtQual"] = test_X["BsmtQual"].fillna("NoBasement")

test_X["BsmtCond"] = test_X["BsmtCond"].fillna("NoBasement")

test_X["BsmtFinType1"] = test_X["BsmtFinType1"].fillna("NoBasement")

test_X["BsmtFinType2"] = test_X["BsmtFinType2"].fillna("NoBasement")



test_X["NotHasFireplace"] = test_X["FireplaceQu"].isnull()

test_X["FireplaceQu"] = test_X["FireplaceQu"].fillna("NoFireplace")



test_X["NotHasGarage"] = test_X["GarageFinish"].isnull()

test_X["GarageFinish"] = test_X["GarageFinish"].fillna("NoGarage")

test_X["GarageQual"] = test_X["GarageQual"].fillna("NoGarage")

test_X["GarageType"] = test_X["GarageType"].fillna("NoGarage")

test_X["GarageCond"] = test_X["GarageCond"].fillna("NoGarage")



test_X["NotHasPool"] = test_X["PoolQC"].isnull()

test_X["PoolQC"] = test_X["PoolQC"].fillna("NoPool")



test_X["NotHasFence"] = test_X["Fence"].isnull()

test_X["Fence"] = test_X["Fence"].fillna("NoFence")



test_X["NotHasKitchen"] = test_X["KitchenQual"].isnull()

test_X["KitchenQual"] = test_X["KitchenQual"].fillna("NoKitchen")



best_pipeline = stacked_model2020

best_pipeline.fit(X, y)

pred_y = best_pipeline.predict(test_X)



output = pd.DataFrame({

    'Id': test_X.index,

    'SalePrice': pred_y

})

output.to_csv('submission.csv', index=False)

!head -n5 submission.csv
show_nulls(test_X)
tft =  test_X

for vals in [tft[col].unique() for col in tft]:

    if None in vals:

        print(vals)
prp = ColumnTransformer(transformers=[

    ('num', SimpleImputer(strategy='mean'), numerical_cols),

    ('const', SimpleImputer(strategy='constant'), categorical_cols)#,

    #('onehot encode', OneHotEncoder(handle_unknown='ignore'), categorical_cols)

])

tft =  pd.DataFrame(prp.fit_transform(test_X))

for vals in [tft[col].unique() for col in tft]:

    print(vals)