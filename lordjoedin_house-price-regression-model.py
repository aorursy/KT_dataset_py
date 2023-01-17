import pandas as pd

pd.options.display.max_columns = 100

pd.options.display.max_rows = 100

import numpy as np

from sklearn.impute import SimpleImputer

import seaborn as sns

import matplotlib.pyplot as plt

import math

import warnings

from IPython.display import set_matplotlib_formats

set_matplotlib_formats('retina')

warnings.filterwarnings('ignore')

%matplotlib inline
train_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
sns.distplot(train_data["SalePrice"]).set_title('Sale Price Distribution Plot')

plt.rcParams["figure.figsize"]=20,10

plt.show()
correlation_matrix_train = train_data.corr()

correlation_matrix_train
sns.heatmap(correlation_matrix_train, annot=False, xticklabels=True, yticklabels=True).set_title('Correlation Matrix')

plt.rcParams["figure.figsize"]=20,20

plt.show()
sales_correlation = correlation_matrix_train["SalePrice"].sort_values()

sales_correlation
low_correlation = [index for index, value in sales_correlation.items() if value < 0.1 and value > -0.1 and index != 'Id']
null_check = train_data.isna().sum()
null_columns = [index for index,value in null_check.items() if value >= train_data.shape[0]*0.8]
train_data.dtypes
catagorical_variables = [index for index, value in train_data.dtypes.items() if value == "object"]
catagorical_variables = [var for var in catagorical_variables if var not in null_columns and var not in low_correlation]
train_data.select_dtypes(include='object').astype('object').describe().transpose()
for i, cat in enumerate(catagorical_variables):

    chart = sns.catplot(x=cat, y="SalePrice", kind="box", data=train_data)

    for ax in chart.axes.flat:

        for label in ax.get_xticklabels():

            label.set_rotation(90)

    plt.show()
train_shape = train_data.shape
combined_data = pd.concat([train_data,test_data])
combined_data.info()
modal_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#Change to mean of certain type of house

combined_data["LotFrontage"] = mean_imputer.fit_transform(combined_data[["LotFrontage"]]) 
#Change to mode of certain type

combined_data["MSZoning"] = modal_imputer.fit_transform(combined_data[["MSZoning"]])

combined_data["Utilities"] = modal_imputer.fit_transform(combined_data[["Utilities"]])

combined_data["Exterior1st"] = modal_imputer.fit_transform(combined_data[["Exterior1st"]])

combined_data["Exterior2nd"] = modal_imputer.fit_transform(combined_data[["Exterior2nd"]])

combined_data["Electrical"] = modal_imputer.fit_transform(combined_data[["Electrical"]])

combined_data["KitchenQual"] = modal_imputer.fit_transform(combined_data[["KitchenQual"]])

combined_data["Functional"] = modal_imputer.fit_transform(combined_data[["Functional"]])

combined_data["GarageType"] = modal_imputer.fit_transform(combined_data[["GarageType"]])

combined_data["SaleType"] = modal_imputer.fit_transform(combined_data[["SaleType"]])
combined_data["MasVnrType"] = combined_data["MasVnrType"].fillna("None")

combined_data["MasVnrArea"] = combined_data["MasVnrArea"].fillna(0)

combined_data["BsmtQual"] = combined_data["BsmtQual"].fillna("No Basement")

combined_data["BsmtCond"] = combined_data["BsmtCond"].fillna("No Basement")

combined_data["BsmtExposure"] = combined_data["BsmtExposure"].fillna("No Basement")

combined_data["BsmtFinType1"] = combined_data["BsmtFinType1"].fillna("No Basement")

combined_data["BsmtFinSF1"] = combined_data["BsmtFinSF1"].fillna(0)

combined_data["BsmtFinType2"] = combined_data["BsmtFinType2"].fillna("No Basement")

combined_data["BsmtUnfSF"] = combined_data["BsmtUnfSF"].fillna(0)

combined_data["BsmtFullBath"] = combined_data["BsmtFullBath"].fillna(0)

combined_data["FireplaceQu"] = combined_data["FireplaceQu"].fillna("No Fireplace")

combined_data["GarageFinish"] = combined_data["GarageFinish"].fillna("No Garage")

combined_data["GarageArea"] = combined_data["GarageArea"].fillna(0)

combined_data["GarageQual"] = combined_data["GarageQual"].fillna("No Garage")

combined_data["GarageCond"] = combined_data["GarageCond"].fillna("No Garage")

combined_data["GarageYrBlt"] = combined_data["GarageYrBlt"].fillna(combined_data["YearBuilt"])

combined_data["GarageCars"] = combined_data["GarageCars"].fillna(0)

combined_data["TotalBsmtSF"] = combined_data["TotalBsmtSF"].fillna(0)
for col in low_correlation:

    del combined_data[col]
for col in null_columns:

    del combined_data[col]
high_correlation = correlation_matrix_train.abs().unstack().sort_values(kind="quicksort", ascending=False)

high_correlation[correlation_matrix_train.shape[0]:].where(lambda x : x >= 0.75).dropna()



# del combined_data["GarageCars"] #high correlation with GarageArea

# del combined_data["GarageYrBlt"] #high correlation with YearBuilt

# del combined_data["TotRmsAbvGrd"] #high correlation with GrLivArea

# del combined_data["TotalBsmtSF"] #high correlation with 1stFlrSF
combined_data.info()
combined_data = pd.get_dummies(combined_data, columns=catagorical_variables, prefix=catagorical_variables, drop_first=True)
new_train = combined_data.iloc[:train_shape[0]]

new_test = combined_data.iloc[train_shape[0]:]
new_train.head()
new_test.head()
X = new_train.drop(['Id', 'SalePrice'], axis=1)

y = new_train["SalePrice"].values

columns = list(X.columns)
test_ids = new_test.iloc[:,0].values

test_set = new_test.iloc[:,1:]

test_set = test_set.drop(['SalePrice'], axis=1)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)

test_set = sc.transform(test_set)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

param_grid_rf = {

    'max_depth': [100, 200, 300],

    'min_samples_leaf': [1, 2, 3],

    'min_samples_split': [1, 2, 3],

    'n_estimators': [1000, 1200]

}

rf_regressor = RandomForestRegressor()

grid_search_rf = GridSearchCV(estimator = rf_regressor, param_grid = param_grid_rf, 

                          cv = 3, n_jobs = -1, verbose = 2)

grid_search_rf.fit(X_train,y_train)
grid_search_rf.best_params_
best_grid_rf = grid_search_rf.best_estimator_

best_grid_rf
# rf_regressor = RandomForestRegressor(n_estimators = 1200,

#                                      max_depth = 200,

#                                      min_samples_leaf = 1,

#                                      min_samples_split = 2,

#                                      random_state=0

#                                     )

rf_regressor = best_grid_rf

rf_regressor.fit(X_train,y_train)
importances = list(rf_regressor.feature_importances_)

feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(columns, importances)]

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

feature_importances
y_pred_rf = rf_regressor.predict(X_test)
from sklearn.metrics import r2_score, mean_squared_error

print(f"R Squared Value: {r2_score(y_test,y_pred_rf)}")
param_grid_xg = {

    'learning_rate': [0.01, 0.02],

    'subsample': [0.6, 0.7, 0.8],

    'colsample_bytree': [0.6, 0.7, 0.8],

    'n_estimators': [800, 1000, 1500, 2000]

}

from xgboost import XGBRegressor

xg_regressor = XGBRegressor()

grid_search_xg = GridSearchCV(estimator = xg_regressor, param_grid = param_grid_xg, 

                          cv = 3, n_jobs = -1, verbose = 2)

grid_search_xg.fit(X_train,y_train)
grid_search_xg.best_params_
best_grid_xg = grid_search_xg.best_estimator_

best_grid_xg
# xg_regressor = XGBRegressor(

#                             learning_rate=0.01, 

#                             n_estimators=1000,

#                             subsample=0.6,

#                             colsample_bytree=0.8

#                            )

xg_regressor = best_grid_xg

xg_regressor.fit(X_train,y_train)
y_pred_xg = xg_regressor.predict(X_test)
print(f"R Squared Value: {r2_score(y_test,y_pred_xg)}")
rf_regressor.fit(X,y)
xg_regressor.fit(X,y)
test_pred_rf = rf_regressor.predict(test_set).round(2)
results_rf = pd.DataFrame(

    {'Id': test_ids,

     'SalePrice': test_pred_rf

    })
results_rf.to_csv("random_forest_prediction.csv", index=False)
test_pred_xg = xg_regressor.predict(test_set).round(2)
results_xg = pd.DataFrame(

    {'Id': test_ids,

     'SalePrice': test_pred_xg

    })
results_xg.to_csv("xgboost_prediction.csv", index=False)