### IMPORTS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, make_scorer
from scipy import stats
import warnings
import copy
from get_smarties import Smarties
warnings.filterwarnings("ignore")

print("Importing dataset...")
dataset = pd.read_csv("train.csv")

### VISUALIZATION
print("Cleaning...")
total = dataset.isnull()
total = total.sum()
total = total.sort_values(ascending=False)
percent = dataset.isnull().count()
percent = (dataset.isnull().sum() / percent)
percent = percent.sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
dataset = dataset.drop((missing_data[missing_data["Total"] > 1]).index, axis=1)
dataset = dataset.drop(dataset.loc[dataset["Electrical"].isnull()].index, axis=0)

dataset = dataset.drop(dataset[dataset["Id"] == 1299].index)
dataset = dataset.drop(dataset[dataset["Id"] == 524].index)

dataset["SalePrice"] = np.log(dataset["SalePrice"])
dataset["GrLivArea"] = np.log(dataset["GrLivArea"])

dataset["HasBsmt"] = pd.Series(len(dataset["TotalBsmtSF"]), index=dataset.index)
dataset["HasBsmt"] = 0
dataset.loc[dataset["TotalBsmtSF"] > 0, "HasBsmt"] = 1
dataset.loc[dataset["HasBsmt"] == 1, "TotalBsmtSF"] = np.log(dataset["TotalBsmtSF"])
dataset = dataset.drop(["HasBsmt"], axis=1)
dataset["MSSubClass"] = dataset["MSSubClass"].map(lambda x : str(x))

dataset["Time"] = 12 * (dataset["YrSold"] - 2006) + dataset["MoSold"]
dataset = dataset.drop(["YrSold", "MoSold"], axis=1)

y = dataset.ix[:, "SalePrice"].values.reshape(-1, 1)
dataset = dataset.drop(["SalePrice"], axis=1)
sm = Smarties()
dataset = sm.fit_transform(dataset)
wanted = [col for col in dataset.columns if col not in ["SalePrice", "Id"]]
X = dataset.ix[:, wanted].values
del((percent, total, missing_data, wanted))

print("Preprocessing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.12)
y_train_org = copy.deepcopy(y_train)
y_val_org = copy.deepcopy(y_val)
# Data preprocessing
scal_ind = [0, 5, 6, 7, 8, 9, 10, 12, 22]
norm_ind = [1, 2, 3, 4, 11, 14, 15, 16, 17, 19, 23, 24, 25, 26, 27, 28, 29, 30]
scal_X = StandardScaler() 
norm_X = Normalizer()
scal_y = StandardScaler()
X_train[:, scal_ind] = scal_X.fit_transform(X_train[:, scal_ind])
X_test[:, scal_ind] = scal_X.transform(X_test[:, scal_ind])
X_val[:, scal_ind] = scal_X.transform(X_val[:, scal_ind])
X_train[:, norm_ind] = norm_X.fit_transform(X_train[:, norm_ind])
X_test[:, norm_ind] = norm_X.transform(X_test[:, norm_ind])
X_val[:, norm_ind] = norm_X.transform(X_val[:, norm_ind])
y_train = scal_y.fit_transform(y_train)
y_test = scal_y.transform(y_test)
y_val = scal_y.transform(y_val)
# SVR
print("Fitting SVR...", end=" ")
from sklearn.svm import SVR
regressor_svr = SVR(C=0.55, epsilon=0.11, kernel="linear")
regressor_svr.fit(X_train, y_train)
y_pred_svr = regressor_svr.predict(X_test)
y_val_svr = regressor_svr.predict(X_val)
print(mean_squared_error(y_test, y_pred_svr))

"""
y_pred_sort = y_pred_svr[y_pred_svr.argsort()]
y_test_sort = y_test[y_test.argsort(axis=0)[:, 0]]
plt.plot(y_test_sort, label="Test")
plt.plot(y_pred_sort, label="Prediction")
plt.legend();
"""
# RandomForest
print("Fitting RFR...", end=" ")
from sklearn.ensemble import RandomForestRegressor
regressor_rfr = RandomForestRegressor(max_features=None, n_estimators=500)
regressor_rfr.fit(X_train, y_train)
y_pred_rfr = regressor_rfr.predict(X_test)
y_val_rfr = regressor_rfr.predict(X_val)
print(mean_squared_error(y_test, y_pred_rfr))

"""
y_pred_sort = y_pred_rfr[y_pred_rfr.argsort()]
y_test_sort = y_test[y_test.argsort(axis=0)[:, 0]]
plt.plot(y_test_sort, label="Test")
plt.plot(y_pred_sort, label="Prediction")
plt.legend()
"""
# XGBRegressor
print("Fitting XGB...", end=" ")
from xgboost.sklearn import XGBRegressor
regressor_xgb = XGBRegressor(n_estimators=300,
                         learning_rate=0.085,
                         booster="dart",
                         objective="reg:linear")
regressor_xgb.fit(X_train, y_train)
y_pred_xgb = regressor_xgb.predict(X_test)
y_val_xgb = regressor_xgb.predict(X_val)
print(mean_squared_error(y_test, y_pred_xgb))

"""
y_pred_sort = y_pred_xgb[y_pred_xgb.argsort()]
y_test_sort = y_test[y_test.argsort(axis=0)[:, 0]]
plt.plot(y_test_sort, label="Test")
plt.plot(y_pred_sort, label="Prediction")
plt.legend()
"""
y_pred_reg = np.concatenate([y_pred_svr.reshape(-1, 1),
                             y_pred_rfr.reshape(-1, 1),
                             y_pred_xgb.reshape(-1, 1)], axis=1)

y_val_reg = np.concatenate([y_val_svr.reshape(-1, 1),
                            y_val_rfr.reshape(-1, 1),
                            y_val_xgb.reshape(-1, 1)], axis=1)
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV

print("Training the ann...", end=" ")
def model_builder():
    estimator = Sequential()
    estimator.add(Dense(units=7, activation="relu", input_dim=len(y_pred_reg[0])))
    estimator.add(Dense(units=5, activation="relu"))
    estimator.add(Dense(units=1))
    estimator.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
    return estimator

regressor = KerasRegressor(build_fn=model_builder, verbose=0, batch_size=3, epochs=85)
regressor.fit(y_pred_reg, y_test)
print(mean_squared_error(y_val, regressor.predict(y_val_reg)))
X_final = np.concatenate((X_train, X_val, X_test), axis=0)
y_final_svr = regressor_svr.predict(X_final)
y_final_rfr = regressor_rfr.predict(X_final)
y_final_xgb = regressor_xgb.predict(X_final)
y_final = np.concatenate((y_final_svr.reshape(-1, 1),
                          y_final_rfr.reshape(-1, 1),
                          y_final_xgb.reshape(-1, 1)), axis=1)
y_final = regressor.predict(y_final)
y_final = np.exp(scal_y.inverse_transform(y_final))
y_final = y_final[y_final.argsort()]
y = y[y.argsort(axis=0)[:, 0]]
y = np.exp(y)
error = mean_squared_error(y, y_final)
print("{:.3E}".format(error))
if error < 1.75e7:
    print("The training was really good! Proceeding...\nImporting test dataset")
    dataset = pd.read_csv("test.csv")
    
    print("Cleaning...")
    total = dataset.isnull()
    total = total.sum()
    total = total.sort_values(ascending=False)
    percent = dataset.isnull().count()
    percent = (dataset.isnull().sum() / percent)
    percent = percent.sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
    del_col = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
               "LotFrontage", "GarageCond", "GarageType", "GarageYrBlt",
               "GarageFinish", "GarageQual", "BsmtExposure", "BsmtFinType2",
               "BsmtFinType1", "BsmtCond", "BsmtQual", "MasVnrArea", "MasVnrType"]
    dataset = dataset.drop(del_col, axis=1)
    dataset = dataset.drop(dataset.loc[dataset["Electrical"].isnull()].index, axis=0)
    
    dataset["GrLivArea"] = np.log(dataset["GrLivArea"])
    dataset["HasBsmt"] = pd.Series(len(dataset["TotalBsmtSF"]), index=dataset.index)
    dataset["HasBsmt"] = 0
    dataset.loc[dataset["TotalBsmtSF"] > 0, "HasBsmt"] = 1
    dataset.loc[dataset["HasBsmt"] == 1, "TotalBsmtSF"] = np.log(dataset["TotalBsmtSF"])
    dataset = dataset.drop(["HasBsmt"], axis=1)
    dataset["MSSubClass"] = dataset["MSSubClass"].map(lambda x : str(x))
    
    dataset["Time"] = 12 * (dataset["YrSold"] - 2006) + dataset["MoSold"]
    dataset = dataset.drop(["YrSold", "MoSold"], axis=1)
    
    dataset = sm.transform(dataset)
    wanted = [col for col in dataset.columns if col not in ["Id"]]
    X = dataset.ix[:, wanted].values
    del((percent, total, missing_data, wanted))

    
    print("Preprocessing...")
    from sklearn.preprocessing import Imputer
    imp = Imputer()
    X = imp.fit_transform(X)
    X[:, scal_ind] = scal_X.transform(X[:, scal_ind])
    X[:, norm_ind] = norm_X.transform(X[:, norm_ind])
    
    y_svr = regressor_svr.predict(X)
    y_rfr = regressor_rfr.predict(X)
    y_xgb = regressor_xgb.predict(X)
    
    y = np.concatenate((y_svr.reshape(-1, 1),
                        y_rfr.reshape(-1, 1),
                        y_xgb.reshape(-1, 1)), axis=1)
    
    predictions = regressor.predict(y)
    predictions = np.exp(scal_y.inverse_transform(predictions))
    
    ids = dataset.ix[:, "Id"]
    
    result = dict()
    result["Id"] = []
    result["SalePrice"] = []
    
    for i in range(len(ids)):
        result["Id"].append(ids[i])
        result["SalePrice"].append(predictions[i])
    
    pd.DataFrame(data=result, columns=["Id", "SalePrice"]).to_csv("results.csv", index=False)