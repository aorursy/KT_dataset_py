import pandas as pd
import numpy as np
import math

import holidays

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, FunctionTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    train_test_split, KFold, GridSearchCV, ParameterGrid,
)
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, DMatrix, plot_importance
from xgboost import cv as xgb_cv
STUDY_START_DATE = pd.Timestamp("2015-01-01 00:00", tz="utc")
STUDY_END_DATE = pd.Timestamp("2020-01-31 23:00", tz="utc")
de_load = pd.read_csv("../input/western-europe-power-consumption/de.csv")
de_load = de_load.drop(columns="end").set_index("start")
de_load.index = pd.to_datetime(de_load.index)
de_load.index.name = "time"
de_load = de_load.groupby(pd.Grouper(freq="h")).mean()
de_load = de_load.loc[
    (de_load.index >= STUDY_START_DATE) & (de_load.index <= STUDY_END_DATE), :
]
de_load.info()
def split_train_test(df, split_time):
    df_train = df.loc[df.index < split_time]
    df_test = df.loc[df.index >= split_time]
    return df_train, df_test

df_train, df_test = split_train_test(
    de_load, pd.Timestamp("2019-02-01", tz="utc")
)
ax = df_train["load"].plot(figsize=(12, 4), color="tab:blue")
_ = df_test["load"].plot(ax=ax, color="tab:orange", ylabel="MW")
df_train.loc[df_train["load"].isna(), :].index
def add_time_features(df):
    cet_index = df.index.tz_convert("CET")
    df["month"] = cet_index.month
    df["weekday"] = cet_index.weekday
    df["hour"] = cet_index.hour
    return df

def add_holiday_features(df):
    de_holidays = holidays.Germany()
    cet_dates = pd.Series(df.index.tz_convert("CET"), index=df.index)
    df["holiday"] = cet_dates.apply(lambda d: d in de_holidays)
    df["holiday"] = df["holiday"].astype(int)
    return df

def add_lag_features(df, col="load"):
    for n_hours in range(24, 49):
        shifted_col = df[col].shift(n_hours, "h")
        shifted_col = shifted_col.loc[df.index.min(): df.index.max()]
        label = f"{col}_lag_{n_hours}"
        df[label] = np.nan
        df.loc[shifted_col.index, label] = shifted_col
    return df

def add_all_features(df, target_col="load"):
    df = df.copy()
    df = add_time_features(df)
    df = add_holiday_features(df)
    df = add_lag_features(df, col=target_col)
    return df

df_train = add_all_features(df_train).dropna()
df_test = add_all_features(df_test).dropna()
df_train.info()
target_col = "load"
X_train = df_train.drop(columns=target_col)
y_train = df_train.loc[:, target_col]
X_test = df_test.drop(columns=target_col)
y_test = df_test.loc[:, target_col]
def fit_prep_pipeline(df):
    cat_features = ["month", "weekday", "hour"]  # categorical features
    bool_features = ["holiday"]  # boolean features
    num_features = [c for c in df.columns
                    if c.startswith("load_lag")]  # numerical features
    prep_pipeline = ColumnTransformer([
        ("cat", OneHotEncoder(), cat_features),
        ("bool", FunctionTransformer(), bool_features),  # identity
        ("num", StandardScaler(), num_features),
    ])
    prep_pipeline = prep_pipeline.fit(df)
    
    feature_names = []
    one_hot_tf = prep_pipeline.transformers_[0][1]
    for i, cat_feature in enumerate(cat_features):
        categories = one_hot_tf.categories_[i]
        cat_names = [f"{cat_feature}_{c}" for c in categories]
        feature_names += cat_names
    feature_names += (bool_features + num_features)
    
    return feature_names, prep_pipeline
feature_names, prep_pipeline = fit_prep_pipeline(X_train)

X_train_prep = prep_pipeline.transform(X_train)
X_train_prep = pd.DataFrame(X_train_prep, columns=feature_names, index=df_train.index)
X_test_prep = prep_pipeline.transform(X_test)
X_test_prep = pd.DataFrame(X_test_prep, columns=feature_names, index=df_test.index)

X_train_prep.info()
lin_model = SGDRegressor(penalty="elasticnet", tol=10, random_state=42)
rf_model = RandomForestRegressor(
    n_estimators=100, criterion='mse', min_samples_leaf=0.001, random_state=42
)
xgb_model = XGBRegressor(n_estimators=1000)
def compute_learning_curves(model, X, y, curve_step, verbose=False):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    n_train_obs = X_train.shape[0]
    n_iter = math.ceil(n_train_obs / curve_step)
    train_errors, val_errors, steps = [], [], []
    for i in range(n_iter):
        n_obs = (i+1) * curve_step
        n_obs = min(n_obs, n_train_obs)
        model.fit(X_train[:n_obs], y_train[:n_obs])
        y_train_predict = model.predict(X_train[:n_obs])
        y_val_predict = model.predict(X_val)
        train_mse = mean_squared_error(y_train[:n_obs], y_train_predict)
        val_mse = mean_squared_error(y_val, y_val_predict)
        train_errors.append(train_mse)
        val_errors.append(val_mse)
        steps.append(n_obs)
        if verbose:
            msg = "Iteration {0}/{1}: train_rmse={2:.2f}, val_rmse={3:.2f}".format(
                i+1, n_iter, np.sqrt(train_mse), np.sqrt(val_mse)
            )
            print(msg)
    return steps, train_errors, val_errors

def plot_learning_curves(steps, train_errors, val_errors, ax=None, title=""):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))
    train_rmse = np.sqrt(train_errors)
    val_rmse = np.sqrt(val_errors)
    ax.plot(steps, train_rmse, color="tab:blue",
            marker=".", label="training")
    ax.plot(steps, val_rmse, color="tab:orange",
            marker=".", label="validation")
    ylim = (0.8*np.median(train_rmse),
            1.5*np.median(val_rmse))
    ax.set_ylim(ylim)
    ax.set_xlabel("Number of observations")
    ax.set_ylabel("RMSE (MW)")
    ax.set_title(title)
    ax.legend()
    ax.grid()
    
steps, train_mse, val_mse = compute_learning_curves(
    lin_model, X_train_prep, y_train, 500, verbose=True
)
plot_learning_curves(steps, train_mse, val_mse, title="Linear model")
rf_steps, rf_train_mse, rf_val_mse = compute_learning_curves(
    rf_model, X_train_prep, y_train, 500, verbose=True
)
plot_learning_curves(rf_steps, rf_train_mse, rf_val_mse, title="Random forest")
xgb_steps, xgb_train_mse, xgb_val_mse = compute_learning_curves(
    xgb_model, X_train_prep, y_train, 500, verbose=True
)
plot_learning_curves(xgb_steps, xgb_train_mse, xgb_val_mse, title="XGB")
xgb_model.fit(X_train_prep, y=y_train)
_, ax = plt.subplots(1, 1, figsize=(6, 6))
_ = plot_importance(xgb_model, ax=ax, max_num_features=30)
def xgb_grid_search_cv(
    params_grid, X, y, nfold,
    num_boost_round=1000, early_stopping_rounds=10,
):
    params_grid = ParameterGrid(params_grid)
    search_results = []
    print(f"Grid search CV : nfold={nfold}, " +
          f"numb_boost_round={num_boost_round}, " +
          f"early_stopping_round={early_stopping_rounds}")
    for params in params_grid:
        print(f"\t{params}")
        cv_df = xgb_cv(
            params=params, dtrain=DMatrix(X, y), nfold=nfold,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            shuffle=False, metrics={"rmse"},
        )
        cv_results = params.copy()
        cv_results["train-rmse-mean"] = cv_df["train-rmse-mean"].min()
        cv_results["test-rmse-mean"] = cv_df["test-rmse-mean"].min()
        search_results.append(cv_results)
    return pd.DataFrame(search_results)
    
params_grid = dict(
    eta = [0.05, 0.1, 0.3],
    max_depth = [2, 4, 6],
    min_child_weight = [5, 1]
)
xgb_search_scores = xgb_grid_search_cv(
    params_grid, X_train_prep, y_train, nfold=4, early_stopping_rounds=10
)
xgb_search_scores.sort_values(by="test-rmse-mean")
final_model = XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=6, min_child_weight=5
)
final_model.fit(
    X_train_prep, y_train, early_stopping_rounds=10,
    eval_set=[(X_train_prep, y_train), (X_test_prep, y_test)],
    verbose=False,
)
final_model.best_score
def compute_predictions_df(model, X, y):
    y_pred = model.predict(X)
    df = pd.DataFrame(dict(actual=y, prediction=y_pred), index=X.index)
    df["squared_error"] =  (df["actual"] - df["prediction"])**2
    return df

pred_df = compute_predictions_df(
    final_model, X_test_prep, y_test
)
pred_df.head()
def plot_predictions(pred_df, start=None, end=None):
    _, ax = plt.subplots(1, 1, figsize=(12, 5))
    start = start or pred_df.index.min()
    end = end or pred_df.index.max()
    pred_df.loc[
        (pred_df.index >= start) & (pred_df.index <= end),
        ["actual", "prediction"]
    ].plot.line(ax=ax)
    ax.set_title("Predictions on test set")
    ax.set_ylabel("MW")
    ax.grid()

plot_predictions(pred_df)
plot_predictions(pred_df,
                 start=pd.Timestamp("2019-04-15", tz="utc"),
                 end=pd.Timestamp("2019-05-06", tz="utc"))
plot_predictions(pred_df,
                 start=pd.Timestamp("2019-12-16", tz="utc"),
                 end=pd.Timestamp("2020-01-06", tz="utc"))
daily_pred_df = pred_df.groupby(pd.Grouper(freq="D")).mean()
daily_pred_df.sort_values(by="squared_error", ascending=False).head(5)
plot_predictions(pred_df,
                 start=pd.Timestamp("2019-06-19", tz="utc"),
                 end=pd.Timestamp("2019-06-22", tz="utc"))
daily_pred_df.sort_values(by="squared_error", ascending=True).head(5)
plot_predictions(pred_df,
                 start=pd.Timestamp("2020-01-23", tz="utc"),
                 end=pd.Timestamp("2020-01-26", tz="utc"))
