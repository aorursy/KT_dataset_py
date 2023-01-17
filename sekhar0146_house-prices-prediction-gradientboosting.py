# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import numpy
import numpy as np
# Import pandas
import pandas as pd
# Import Ridge from sklearn's GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
# Import Ridge from sklearn's RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
# Import Ridge from sklearn's linear_model module
from sklearn.linear_model import Ridge
# Import SVR from sklearn's svm module
from sklearn.svm import SVR

pd.set_option('display.max_rows', None)

# Import the data
df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
# print(df.head())
# print(df.shape)
# print(df.isna().sum())
# print(df.dtypes)

# Make a copy of original dataframe - for future reference
df_tmp = df.copy()

def preprocess_data(df):
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                #df[label+"_is_missing"] = pd.isnull(content)
                df[label] = content.fillna(content.median())
                
        if not pd.api.types.is_numeric_dtype(content):
            #df[label+"_is_missing"] = pd.isnull(content)
            df[label] = pd.Categorical(content).codes+1
    return df

# Process train data
df_train = preprocess_data(df_tmp)

np.random.seed(42)
# Split into X and y (on train set)
X_train = df_train.drop("SalePrice", axis=1)
y_train = df_train["SalePrice"]

# Import the test ata
df_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

# Process test data
df_test = preprocess_data(df_test)

# ===========================================
# Modele processing 
# ===========================================
models={"GradientBoostingRegressor": GradientBoostingRegressor(),
        "RandomForestRegressor": RandomForestRegressor(),
        "Ridge": Ridge(),
        "SVR_linear": SVR(kernel="linear"),
        "SVR_rbf": SVR(kernel="rbf")
        }
reg_result={}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    reg_result[model_name] = model.score(X_train, y_train)

print(reg_result)

# -----------------------------------
# calculate RMSLE
# -----------------------------------
from sklearn.metrics import mean_squared_log_error

def show_rlsle(model):
    train_preds = model.predict(X_train)
    rmsle = np.sqrt(mean_squared_log_error(y_train, train_preds))
    return rmsle

# ------------------------------------------------------------
# Hyperparameter tuning with RandomizedSearchCV
# ------------------------------------------------------------
print("=== Hyperparameter tuning with RandomForestRegressor ===")
from sklearn.model_selection import RandomizedSearchCV
# different RandomForestRegressor hyperparameters
rf_rf_grid = {"n_estimators": np.arange(200, 2000, 10),
           "max_depth": [None, 3 , 5, 10, 20, 30],
           "min_samples_split":np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2),
           "max_features": [0.5, 1, "sqrt", "auto"]
           }

# Instantiate RandomizedSearchCV model
rs_rf_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,
                                                       random_state=42),
                              param_distributions=rf_rf_grid,
                              n_iter=2,
                              cv=5,
                              verbose=True)
rs_rf_model.fit(X_train, y_train)
print("Hyperparameter best parameters for RandomForestRegressor ==>")
print(rs_rf_model.best_params_)
print(show_rlsle(rs_rf_model))
print("RandomForestRegressor score : ", rs_rf_model.score(X_train, y_train))
print("")   
"""
Hyperparameter best parameters for RandomForestRegressor ==>
{'n_estimators': 1700, 'min_samples_split': 8, 'min_samples_leaf': 7, 'max_features': 'auto', 'max_depth': 10}
0.11026668785556953
RandomForestRegressor score :  0.917946936801314

"""
print("=== Hyperparameter tuning with GradientBoostingRegressor ===")
# different GradientBoostingRegressor hyperparameters
rf_gb_grid = {"n_estimators": np.arange(200, 2000, 10),
           "max_depth": [None, 3 , 5, 10, 20, 30],
           "min_samples_split":np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2),
           "max_features": [0.5, 1, "sqrt", "auto"],
           "learning_rate": [0.1, 0.05, 0.02, 0.01]
           }

# Instantiate GradientBoostingRegressor model
rs_gb_model = RandomizedSearchCV(GradientBoostingRegressor(random_state=42),
                              param_distributions=rf_gb_grid,
                              n_iter=2,
                              cv=5,
                              verbose=True)
rs_gb_model.fit(X_train, y_train)
print("Hyperparameter best parameters for GradientBoostingRegressor ==>")
print(rs_gb_model.best_params_)
print(show_rlsle(rs_gb_model))
print("GradientBoostingRegressor score : ", rs_gb_model.score(X_train, y_train))
print("")
"""
Hyperparameter best parameters for GradientBoostingRegressor ==>
{'n_estimators': 830, 'min_samples_split': 12, 'min_samples_leaf': 11, 'max_features': 'auto', 'max_depth': 30, 'learning_rate': 0.02}
0.017677890424582995
GradientBoostingRegressor score :  0.9979757850646321
"""

# As per the hyperparameter tuning, understand that GradientBoostingRegressor model is giving most accurate results
# ---------------------------------------------------------------
# Make predictions on test data set on GradientBoostingRegressor
# ---------------------------------------------------------------
print("========= Predicted SalePrice ============ ")
test_pred = rs_gb_model.predict(df_test)
print(test_pred)
print(test_pred.shape)
print("")

# Format as per requirement
df_prediction_saleprice = pd.DataFrame()
df_prediction_saleprice["Id"] = df_test["Id"]
df_prediction_saleprice["SalePrice"] = test_pred
print(df_prediction_saleprice.head())

# Save the prediction results in the csv file

