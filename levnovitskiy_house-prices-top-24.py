import numpy as np

import pandas as pd
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')  #traning data

y = pd.DataFrame(train["SalePrice"])  #target

train = train.drop(labels=["SalePrice"], axis=1)  #traning set

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')  #testing data

ids = test["Id"]  #array of test ids (further needed for submission)
train.head()
y.head()
test.head()
cat_features = ["MSZoning", "MSSubClass", "Street", "Alley",

                "LotShape", "LandContour", "Utilities", "LotConfig",

                "LandSlope", "Neighborhood","Condition1", 

                "Condition2", "BldgType", "HouseStyle", 

                "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",

                "MasVnrType", "ExterQual", "ExterCond", "Foundation",

                "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",

                "BsmtFinType2", "Heating", "HeatingQC", "CentralAir",

                "Electrical", "Functional", "FireplaceQu", "GarageType",

                "GarageFinish", "GarageQual", "GarageCond", "PavedDrive",

                "SaleType", "SaleCondition", "PoolQC", "Fence", "MiscFeature",

                "KitchenQual"] 
ordinal = ["OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", 

 "GarageYrBlt", "GarageCars", "MoSold", "YrSold"]
absolute = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", 

 "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",

 "LowQualFinSF", "BsmtFullBath", "BsmtHalfBath", "FullBath", 

 "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", 

 "Fireplaces", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",

 "3SsnPorch", "ScreenPorch", "PoolArea"]
train[absolute].isnull().sum()
train[cat_features].isnull().sum()
train[ordinal].isnull().sum()
!pip install category_encoders
import category_encoders as ce

cb_enc = ce.CatBoostEncoder(cols=cat_features)



cb_enc.fit(train[cat_features], y)  #fit the encoder



train = train.drop(labels=cat_features, axis=1).join(cb_enc.transform(train[cat_features]))  #transform feature columns

test = test.drop(labels=cat_features, axis=1).join(cb_enc.transform(test[cat_features]))

print(len(test))
train.head(3)
from sklearn.impute import KNNImputer

knn_imp = KNNImputer()



train = train.drop(labels=absolute, axis=1).join(pd.DataFrame(knn_imp.fit_transform(train[absolute]),

                   columns=absolute))

test = test.drop(labels=absolute, axis=1).join(pd.DataFrame(knn_imp.fit_transform(test[absolute]),

                   columns=absolute))
train[absolute].isnull().sum()
train = train.drop(labels=ordinal, axis=1).join(pd.DataFrame(knn_imp.fit_transform(train[ordinal]),

                   columns=ordinal))

test = test.drop(labels=ordinal, axis=1).join(pd.DataFrame(knn_imp.fit_transform(test[ordinal]),

                   columns=ordinal))
train[ordinal].isnull().sum()
dict(train.isnull().sum())
dict(train.dtypes)
print("Quantity of features:", f"train: {len(train.columns)}", f"test: {len(test.columns)}", sep="\n")
from sklearn.feature_selection import SelectKBest, f_classif  # import the needed classes



columns = train.columns

index_train = train.index

index_test = test.index



selector = SelectKBest(f_classif, k=70)

selector.fit(train, y)



train = selector.transform(train)

test = selector.transform(test)



train, test = (

    pd.DataFrame(selector.inverse_transform(train),

                   index=index_train,

                   columns=columns), 

    pd.DataFrame(selector.inverse_transform(test), 

                   index=index_test,

                   columns=columns)

              )



selected_columns = train.columns[train.var() != 0]



# Get the valid dataset with the selected features.

train = train[selected_columns]

test = test[selected_columns]



print("Quantity of features:", f"train: {len(train.columns)}", f"test: {len(test.columns)}", sep="\n")
!pip install catboost
from catboost import CatBoostRegressor

cbr = CatBoostRegressor(

    eval_metric="RMSE"

)



cbr.fit(

    train,

    y

)
y_pred = cbr.predict(test)
submission = pd.DataFrame({

        "Id": ids,

        "SalePrice": y_pred

    })

submission.to_csv('submission.csv', index=False)