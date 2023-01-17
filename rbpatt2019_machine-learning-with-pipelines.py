import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("../input/train.csv")
Y = data.SalePrice
X = data.drop(["SalePrice"], axis=1)

test = pd.read_csv("../input/test.csv")

X.describe()
#MSSubClass is actually categorical, not numerical, but NaN cannot be deduced from logic
X.MSSubClass = X.MSSubClass.astype("O")
test.MSSubClass = test.MSSubClass.astype("O")
#ExterQual, ExterCond, BsmtQual, BsmtCond, BsmtExposure
#BsmtFinType1, BsmtFinType2, HeatingQC, KitchenQual
#Functional, FirePlaceQu, GarageFinish, GarageQual 
#GarageCond, PavedDrive, PoolQC are object but has numerical relation (ex > gd)
X.replace({"ExterQual": {"Ex": 5,
                         "Gd": 4,
                         "TA": 3,
                         "Fa": 2,
                         "Po": 1
                        },
           "ExterCond": {"Ex": 5,
                         "Gd": 4,
                         "TA": 3,
                         "Fa": 2,
                         "Po": 1
                        },
           "BsmtQual": {"Ex": 5,
                        "Gd": 4,
                        "TA": 3,
                        "Fa": 2,
                        "Po": 1,
                        np.nan: 0
                        },
           "BsmtCond": {"Ex": 5,
                        "Gd": 4,
                        "TA": 3,
                        "Fa": 2,
                        "Po": 1,
                        np.nan: 0
                        },
           "BsmtExposure": {"Gd": 4,
                            "Av": 3,
                            "Mn": 2,
                            "No": 1,
                            np.nan: 0
                            },
           "BsmtFinType1": {"GLQ": 6,
                            "ALG": 5,
                            "BLQ": 4,
                            "Rec": 3,
                            "LwQ": 2,
                            "Unf": 1,
                            np.nan: 0
                            },
           "BsmtFinType2": {"GLQ": 6,
                            "ALG": 5,
                            "BLQ": 4,
                            "Rec": 3,
                            "LwQ": 2,
                            "Unf": 1,
                            np.nan: 0
                            },
           "HeatingQC": {"Ex": 5,
                         "Gd": 4,
                         "TA": 3,
                         "Fa": 2,
                         "Po": 1
                        },
           "KitchenQual": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1,
                            np.nan: 0
                        },
           "Functional": {"Typ": 7,
                          "Min1": 6,
                          "Min2": 5,
                          "Mod": 4,
                          "Maj1": 3,
                          "Maj2": 2,
                          "Sev": 1,
                          "Sal": 0
                          },
           "FireplaceQu": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1,
                            np.nan: 0
                          },
           "GarageFinish": {"Fin": 3,
                            "RFn": 2,
                            "Unf": 1,
                            np.nan: 0
                          },
           "GarageQual": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1,
                            np.nan: 0
                          },
           "GarageCond": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1,
                            np.nan: 0
                          },
           "PavedDrive": {"Y": 3,
                            "P": 2,
                            "N": 1,
                            np.nan: 0
                          },
           "PoolQC": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1,
                            np.nan: 0
                          }
          }, inplace=True)
test.replace({"ExterQual": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1
                           },
              "ExterCond": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1
                           },
              "ExterCond": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1,
                            np.nan: 0
                           },
           "BsmtQual": {"Ex": 5,
                        "Gd": 4,
                        "TA": 3,
                        "Fa": 2,
                        "Po": 1,
                        np.nan: 0
                        },
              "BsmtCond": {"Ex": 5,
                           "Gd": 4,
                           "TA": 3,
                           "Fa": 2,
                           "Po": 1,
                           np.nan: 0
                          },
           "BsmtExposure": {"Gd": 4,
                            "Av": 3,
                            "Mn": 2,
                            "No": 1,
                            np.nan: 0
                            },
           "BsmtFinType1": {"GLQ": 6,
                            "ALG": 5,
                            "BLQ": 4,
                            "Rec": 3,
                            "LwQ": 2,
                            "Unf": 1,
                            np.nan: 0
                            },
           "BsmtFinType2": {"GLQ": 6,
                            "ALG": 5,
                            "BLQ": 4,
                            "Rec": 3,
                            "LwQ": 2,
                            "Unf": 1,
                            np.nan: 0
                            },
           "HeatingQC": {"Ex": 5,
                         "Gd": 4,
                         "TA": 3,
                         "Fa": 2,
                         "Po": 1
                        },
           "KitchenQual": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1,
                            np.nan: 0
                        },
           "Functional": {"Typ": 7,
                          "Min1": 6,
                          "Min2": 5,
                          "Mod": 4,
                          "Maj1": 3,
                          "Maj2": 2,
                          "Sev": 1,
                          "Sal": 0
                          },
           "FireplaceQu": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1,
                            np.nan: 0
                          },
           "GarageFinish": {"Fin": 3,
                            "RFn": 2,
                            "Unf": 1,
                            np.nan: 0
                          },
           "GarageQual": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1,
                            np.nan: 0
                          },
           "GarageCond": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1,
                            np.nan: 0
                          },
           "PavedDrive": {"Y": 3,
                            "P": 2,
                            "N": 1,
                            np.nan: 0
                          },
           "PoolQC": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1,
                            np.nan: 0
                          }
          }, inplace=True)
XClean = X.select_dtypes(exclude=["O"])
#XClean.isna().sum() to see where NaN are
XClean.fillna(0, inplace=True)
testClean = test.select_dtypes(exclude=["O"])
#testClean.isna().sum() to see where NaN are
testClean.fillna(0, inplace=True)
pipeline = Pipeline([#imputer 
                     #encoder
                     #feature select or reduction
                     ("model", XGBRegressor(random_state=0))
                    ])

trainX, testX, trainY, testY = train_test_split(XClean, Y)
paramGrid = {
    "model__n_estimators": [10, 50, 100, 250, 500, 750, 1000],
    "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1]
}

modelCV = GridSearchCV(pipeline, 
                       cv=5, 
                       param_grid=paramGrid)
modelCV.fit(trainX, trainY)
modelCV.best_params_
modelCV.refit
print(mean_absolute_error(testY, modelCV.predict(testX)))
pipeline = Pipeline([#imputer 
                     #encoder
                     #feature select or reduction
                     ("model", XGBRegressor(random_state=0,
                                            learning_rate=0.1,
                                            n_estimators=750
                                           ))
                    ])
pipeline.fit(XClean, Y)
output = pd.DataFrame({"Id": testClean.Id,
                       "SalePrice": pipeline.predict(testClean)
                      }
                     ).set_index("Id")
output.to_csv("submission.csv")
