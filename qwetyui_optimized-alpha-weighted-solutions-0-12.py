import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import sklearn
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.base import clone
from sklearn.kernel_ridge import KernelRidge
import warnings
from IPython.display import clear_output
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
ROOT_PATH = "../input/"
TRAIN_FILE = ROOT_PATH + "train.csv"
TEST_FILE = ROOT_PATH + "test.csv"
TEST_SIZE = 0.05
SUBM_FILE = "res"
DROPOUT_USELESS_COLS = 3
import math
LOGBASE = math.e

def custlog(arr):
    return np.log(arr) / np.log(LOGBASE)

def custexp(arr):
    return LOGBASE ** arr

def priceencode(arr):
    return custlog(arr) - 30

def pricedecode(arr):
    return custexp(arr + 30)
import math
def prepare(df):
    df = df.drop(["Id"], axis=1)
    df['MSZoning'] = df['MSZoning'].apply(lambda x: 'RL' if x == None else x)
    df['Exterior2nd'] = df['Exterior2nd'].apply(lambda x: 'VinylSd' if x == None else x)
    df['SaleType'] = df['SaleType'].apply(lambda x: 'WD' if x == None else x)
    df['SaleCondition'] = df['SaleCondition'].apply(lambda x: 'Normal' if x == None else x)
    for column in df.columns:
        if type(df[column][0]) in [str] or str(df[column][0]) == "nan":
            nf = pd.get_dummies(df[column], prefix=column)
            for col in nf:
                df[col] = nf[col]
            df = df.drop([column], axis=1)
    
    for column in df.columns:
        if type(df[column][0]) in [np.int64, np.float64]:
            df[column] = df[column].apply(lambda x: df[column].mean() if str(x) == "nan" else x)
    
    df["LotFrontage"] = df["LotFrontage"].apply(lambda x: 1 if x > 60 else 0)
    df["MSSubClass"] = df["MSSubClass"].apply(lambda x: x**0.5)
    df["TotRmsAbvGrd"] = df["TotRmsAbvGrd"].apply(lambda x: x**2 / 30)
    if "SalePrice" in df.columns:
        df["SalePrice"] = df["SalePrice"].apply(priceencode)
    years = ["YearBuilt", "YearRemodAdd", "YrSold"]
    for col in years:
        df[col] = df[col].apply(lambda x: np.log(2020 - x))
    tolog = ["WoodDeckSF", "3SsnPorch", "1stFlrSF", "2ndFlrSF", "TotalBsmtSF", "TotalBsmtSF"]
    for col in tolog:
        df[col] = df[col].apply(lambda x: np.log(x+1))
    return df.astype(np.float64)
def fix_empties(df, cols):
    for col in cols:
        if not col in df.columns:
            df.insert(0, col, 0)
def __sync__(df_full, df_part):
    to_fix = []
    for col in df_full:
        if not col in df_part.columns:
            to_fix.append(col)
    fix_empties(df_part, to_fix)

def synchronize(df1, df2):
    __sync__(df2, df1)
    __sync__(df1, df2)
    df1 = df1.reindex_axis(sorted(df1.columns), axis=1)
    df2 = df2.reindex_axis(sorted(df2.columns), axis=1)
    return df1, df2
raw_train_data = prepare(pd.read_csv(TRAIN_FILE))
raw_test_data = prepare(pd.read_csv(TEST_FILE))
raw_train_data, raw_test_data = synchronize(raw_train_data, raw_test_data)
def check_if_synced(data1, data2):
    assert len(data1.columns) == len(data2.columns), "Data column count does not match!"
    for i in range(len(data1.columns)):
        assert data1.columns[i] == data1.columns[i], "Columns don't match!"
check_if_synced(raw_train_data, raw_test_data)
sns.distplot(raw_train_data["SalePrice"])
raw_train_data.head(90)
def visualize(data, classes):
    for col in classes:
        fig, ax = plt.pyplot.subplots()
        ax.scatter(x = data[col], y = data['SalePrice'])
        plt.pyplot.ylabel('SalePrice', fontsize=13)
        plt.pyplot.xlabel(col, fontsize=13)
        plt.pyplot.show()

visualize(raw_train_data, ["MSSubClass", "TotRmsAbvGrd", "YearBuilt", "YearRemodAdd", "YrSold", "WoodDeckSF", "3SsnPorch", "1stFlrSF", "2ndFlrSF", "TotalBsmtSF"])
def column_mean_ignore(data, col, num_to_ignore):
    s = 0
    c = 0
    for n in data[col]:
        if n != num_to_ignore:
            s += n
            c += 1
    return s / c

def final_prepare(data):
    useless_columns = ["YrSold", "MSSubClass"]
    for useless_column in useless_columns:
        data = data.drop([useless_column], axis=1)
    mean_columns = ["TotalBsmtSF", "2ndFlrSF", "3SsnPorch", "WoodDeckSF"]
    for mean_column in mean_columns:
        m = column_mean_ignore(data, mean_column, 0)
        data[mean_column] = data[mean_column].apply(lambda x: x if x != 0 else m)
    data["TotalSF"] = data["TotalBsmtSF"] + data["1stFlrSF"] + data["2ndFlrSF"]
    offsets = {"1stFlrSF": -7, "2ndFlrSF": -6.5, "3SsnPorch": -5.2}
    for key in offsets:
        data[key] = data[key].apply(lambda x: x + offsets[key])
    return data
    
train_data, test_data = final_prepare(raw_train_data), final_prepare(raw_test_data)
check_if_synced(train_data, test_data)
visualize(train_data, ["TotalSF", "TotRmsAbvGrd", "YearBuilt", "YearRemodAdd", "WoodDeckSF", "3SsnPorch", "1stFlrSF", "2ndFlrSF", "TotalBsmtSF"])
train_data.head(90)
prices = pd.DataFrame({"price":train_data["SalePrice"]})
prices.hist()
import math
def metric(y_pred, y_true):
    return (((y_pred) - (y_true))**2).mean()**0.5, np.abs(np.exp(y_pred) - np.exp(y_true)).mean()
if DROPOUT_USELESS_COLS != 0:
    errors = []
    for col in train_data.drop(["SalePrice"], axis=1).columns:
        X = train_data.drop(["SalePrice"], axis=1).drop([col], axis=1)
        y = train_data["SalePrice"]
        X = np.array(X)
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
        m = Lasso(alpha=0.001)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        errors.append((metric(y_pred, y_test)[0], col))
if DROPOUT_USELESS_COLS != 0:
    todrop = [g[1] for g in sorted(errors, key=(lambda x: x[0]))[:DROPOUT_USELESS_COLS]]
    train_data = train_data.drop(todrop, axis=1)
    test_data = test_data.drop(todrop, axis=1)
X = train_data.drop(["SalePrice"], axis=1)
y = train_data["SalePrice"]
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
class WeightedAvgRegressor:
    def __init__(self, models):
        self.models = models
    
    def fit(self, X, y, X_test, y_test, graphic=True):
        self.models_ = [{"model": clone(x), "lrmse": 0} for x in self.models]
        
        id = 0
        errs = []
        for model in self.models_:
            model["model"].fit(X, y)
            model["lrmse"] = metric(model["model"].predict(X_test), y_test)[0]
            id += 1
            if not graphic:
                print("#", id, "fitted with lrmse =", model["lrmse"])
            else:
                errs.append(metric(self.predict(X_test), y_test)[0])
                clear_output(True)
                plt.pyplot.plot(errs, label="lrmse")
                plt.pyplot.show()
        r = metric(self.predict(X_test), y_test)[0]
        print("Done. General error =", r)
        return r

    def predict(self, X):
        predictions = 0.0
        errs = 0
        for model in self.models_:
            if model["lrmse"] != 0:
                form = 1 / (model["lrmse"] ** 2)
                errs += form
                predictions += model["model"].predict(X) * form
        return predictions / errs
class AlphaOptimizer:
    def __init__(self, modeloptlist):
        self.mol = modeloptlist
        self.opt = None
        self.war = None
    
    def optim_model(self, fiter, X, y, X_test, y_test):
        deltas = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.85, 1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 7, 12, 16, 30, 50]
        errors = []
        for delta in deltas:
            fitted_model = fiter(X, y, delta)
            y_pred = fitted_model.predict(X_test)
            y_true = y_test
            error = metric(y_pred, y_true)[0]
            errors.append(error)
        min_id = 0
        for err_id in range(len(errors)):
            if errors[err_id] < errors[min_id]:
                min_id = err_id
        return deltas[min_id], errors, deltas
    
    def optim(self, X, y, X_test, y_test, output=False):
        whole = {}
        mins = []
        for m in self.mol:
            minimum, diagram, deltas = self.optim_model(m[0], X, y, X_test, y_test)
            if output:
                plt.pyplot.plot(deltas, diagram, label="Errors for " + m[1])
                plt.pyplot.legend()
                plt.pyplot.show()
            whole[m[1]] = diagram
            mins.append(minimum)
        
        return mins
        
    def fit(self, X, y, X_test, y_test, graphic=False):
        if self.opt == None:
            self.opt = self.optim(X, y, X_test, y_test)
        models = []
        id = 0
        for m in self.mol:
            models.append(m[0](X, y, self.opt[id]))
            id += 1
        war = WeightedAvgRegressor(models)
        war.fit(X, y, X_test, y_test, graphic=graphic)
        self.war = war
        return war
    
    def predict(self, X):
        assert self.war != None, "No weighted average regressor found! Run ao.fit() to init WAR!"
        return self.war.predict(X)
def fiter_lasso(X, y, delta):
    m = Lasso(alpha=0.0005 * delta, max_iter=2000)
    m.fit(X, y)
    return m

def fiter_Bay(X, y, delta):
    m = BayesianRidge(500, tol=0.06200 * delta)
    m.fit(X, y)
    return m

def fiter_xgb(X, y, delta):
    m = XGBRegressor(max_depth=4, learning_rate=0.5 * delta, n_estimators=400)
    m.fit(X, y)
    return m

def fiter_gbr(X, y, delta):
    m = GradientBoostingRegressor(learning_rate=0.01 * delta, max_depth=4, n_estimators=400)
    m.fit(X, y)
    return m

def fiter_lgb(X, y, delta):
    m = LGBMRegressor(learning_rate=0.04 * delta, max_depth=4)
    m.fit(X, y)
    return m

def fiter_cat(X, y, delta):
    m = CatBoostRegressor(iterations=1200, learning_rate=0.5 * delta, depth=4, logging_level="Silent")
    m.fit(X, y)
    return m

def fiter_ridge(X, y, delta):
    m = KernelRidge(alpha=0.05 * delta)
    m.fit(X, y)
    return m
ao = AlphaOptimizer(
    [
        [fiter_lasso, "lasso1"],
        [fiter_lasso, "lasso2"],
        [fiter_Bay,   "bay1"],
        [fiter_xgb,   "xgb1"],
        [fiter_gbr,   "gbr1"],
        [fiter_lgb,   "lgb1"],
        [fiter_ridge, "ridge1"]
    ]
)
ao.optim(X_train, y_train, X_test, y_test, output=True)
ao.fit(X_train, y_train, X_test, y_test, True)
print(metric(ao.predict(X_test), y_test))
err = metric(ao.predict(X_test), y_test)
print("RMSE:", err[0])
print("exp. RMSE:", err[1], "(linear error)")
X_to_predict = np.array(test_data.drop(["SalePrice"], axis=1))
y_commit = ao.predict(X_to_predict)
sns.distplot(pricedecode(y_commit))
f = open(SUBM_FILE + "_" + str(round(err[0], 5)) + "_.csv", "wt")
f.write('Id,SalePrice\n')
i = 0
for y in list(y_commit):
    f.write(str(i + 1461) + "," + str(pricedecode(y)) + "\n")
    i += 1
f.close()