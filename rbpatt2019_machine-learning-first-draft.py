import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

data = pd.read_csv("../input/train.csv")
Y = data.SalePrice
X = data.drop(["SalePrice"], axis=1)

test = pd.read_csv("../input/test.csv")

X.describe()
(X
 .isna()
 .sum()
 .pipe(lambda series: series[series > 0])
)
"""
Because of the nature of this data and the descriptions provided
the value of most of the missing data can be inferred without imputation
Lot frontage will be NaN if there is no street, so replace na with 0
Alley will be NaN if there is no alley entrance, so repalce with "None"
While MasVnrType does have a "none" value that is different from NaN,
I think it makes the msot sense to replace these with none as well
MasVnrArea will be 0 if NaN is present
Same goes for all the Bsmt columns
Houses have to have electrical, so replace unknown
Fireplace NaN is None
same for all garage entries (YrBlt as 0)
same for pool and fence and miscFeature
"""
XClean = X.fillna(value={"LotFrontage": 0.0,
                         "Alley": "None",
                         "MasVnrType": "None",
                         "MasVnrArea": 0.0,
                         "BsmtQual": "None",
                         "BsmtCond": "None",
                         "BsmtExposure": "None",
                         "BsmtFinType1": "None",
                         "BsmtFinType2": "None",
                         "FireplaceQu": "None",
                         "GarageType": "None",
                         "GarageYrBlt": 0.0,
                         "GarageFinish": "None",
                         "GarageQual": "None",
                         "GarageCond": "None",
                         "PoolQC": "None",
                         "Fence": "None",
                         "MiscFeature": "None"
                        }
                 )
XClean.isna().sum().pipe(lambda series: series[series > 0])
(test
 .isna()
 .sum()
 .pipe(lambda series: series[series > 0])
)
"""
A number of these are the same and can easily be addressed with the same logic as above
The bsmt columns new to the test data are all linked to an entry with BsmtQual == NaN
so can be set to none/0.0
"""
testClean = test.fillna(value={"LotFrontage": 0.0,
                         "Alley": "None",
                         "MasVnrType": "None",
                         "MasVnrArea": 0.0,
                         "BsmtQual": "None",
                         "BsmtCond": "None",
                         "BsmtExposure": "None",
                         "BsmtFinType1": "None",
                         "BsmtFinSF1": 0.0,
                         "BsmtFinType2": "None",
                         "BsmtFinSF2": 0.0,
                         "BsmtUnfSF": 0.0,
                         "TotalBsmtSF" : 0.0,
                         "BsmtFullBath" : 0.0,
                         "BsmtHalfBath" : 0.0,
                         "FireplaceQu": "None",
                         "GarageType": "None",
                         "GarageYrBlt": 0,
                         "GarageFinish": "None",
                         "GarageQual": "None",
                         "GarageCond": "None",
                         "PoolQC": "None",
                         "Fence": "None",
                         "MiscFeature": "None"
                        }
                 )
testClean.isna().sum().pipe(lambda series: series[series > 0])
"""
The remaining values cannot be deduced from trends, so a simple imputer will be used to fill them in
For numeric values, the mean will be used. For objects, the most frequent will be
First we train_test_split for XGBoost
"""
trainX, evalX, trainY, evalY = train_test_split(XClean, Y, test_size=0.2)

trainXNumeric = trainX.select_dtypes(include=["int64", "float64"])
evalXNumeric = evalX.select_dtypes(include=["int64", "float64"])
imputeNumeric = SimpleImputer(strategy="mean")
imputeNumeric.fit_transform(trainXNumeric)
imputeNumeric.transform(evalXNumeric)

trainXObject = trainX.select_dtypes(include=["object"])
evalXObject = evalX.select_dtypes(include=["object"])
imputeObject = SimpleImputer(strategy="most_frequent")
imputeObject.fit_transform(trainXObject)
imputeObject.transform(evalXObject)
trainXImputed = trainXNumeric.join(trainXObject)
evalXImputed = evalXNumeric.join(trainXObject)
for col in trainXImputed.columns:
    if XClean[col].dtype == "O":
        print("Column: %s %d  " % (col, len(trainXImputed[col].unique())))
#Neighborhood is on the higher side, but will use it to prevent data leakage.
trainXCoded = pd.get_dummies(trainXImputed)
evalXCoded = pd.get_dummies(evalXImputed)
trainXFinal, evalXFinal = trainXCoded.align(evalXCoded,
                                            join="inner",
                                            axis=1
                                           )
model = XGBRegressor(n_estimators=1000,
                     learning_rate=0.01,
                     random_state=0
                    )
model.fit(trainXFinal, trainY,
          early_stopping_rounds=5,
          eval_set=[(evalXFinal, evalY)]
         )
XNumeric = XClean.select_dtypes(include=["int64", "float64"])
testNumeric = testClean.select_dtypes(include=["int64", "float64"])
imputeNumeric = SimpleImputer(strategy="mean")
imputeNumeric.fit_transform(XNumeric)
imputeNumeric.transform(testNumeric)

XObject = XClean.select_dtypes(include=["object"])
testObject = testClean.select_dtypes(include=["object"])
imputeObject = SimpleImputer(strategy="most_frequent")
imputeObject.fit_transform(XObject)
imputeObject.transform(testObject)

XImputed = XNumeric.join(XObject)
testImputed = testNumeric.join(testObject)

XCoded = pd.get_dummies(XImputed)
testCoded = pd.get_dummies(testImputed)
XFinal, testFinal = XCoded.align(testCoded,
                                 join="inner",
                                 axis=1
                                )

model = XGBRegressor(n_estimators=555,
                     learning_rate=0.01,
                     random_state=0
                    )
model.fit(XFinal, Y)
model.predict(testFinal)
output = pd.DataFrame({"Id": testFinal.Id,
                       "SalePrice": model.predict(testFinal)
                      }
                     ).set_index("Id")
output.to_csv("submission.csv")
