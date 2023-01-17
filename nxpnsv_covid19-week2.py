import os

import numpy as np

import pandas as pd

from scipy.optimize.minpack import curve_fit
def load_kaggle_csv(dataset: str) -> pd.DataFrame:

    df = pd.read_csv(f"/kaggle/input/covid19-global-forecasting-week-2/{dataset}.csv", parse_dates=["Date"])

    df["Province_State"].fillna("", inplace=True)

    df["DayOfYear"] = df["Date"].dt.dayofyear

    df["Date"] = df["Date"].dt.date

    return df
train = load_kaggle_csv("train")

test = load_kaggle_csv("test")
print(train.head())

print(train.describe())
def RMSLE(prediction, actual):

    return np.sqrt(

        np.mean(np.power(np.log1p(prediction) - np.log1p(actual), 2))

    )





def logistic(x, x0, L, k):

    return L / (1 + np.exp(-k * (x - x0)))





def fit(function, x, y, maxfev):

    # Fuzzy fitter

    p0 = [np.median(x), y[-1], 0.1]

    pn0 = p0 * (np.random.random(len(p0)) + [0.5, 1.0, 0.5])

    try:

        params, pcov = curve_fit(

            function,

            x,

            y,

            p0=pn0,

            maxfev=maxfev,

            sigma=np.maximum(1, np.sqrt(y)) * (0.1 + 0.9 * np.random.random()),

            bounds=([0, y[-1], 0.01], [200, 1e6, 1.5]),

        )

        pcov = pcov[np.triu_indices_from(pcov)]

    except (RuntimeError, ValueError):

        params = p0

        pcov = np.zeros(len(p0) * (len(p0) - 1))

    y_hat = function(x, *params)

    rmsle = RMSLE(y_hat, y)

    return (params, pcov, rmsle, y_hat)





def fit_model(df: pd.DataFrame, n_samples=2, maxfev=1000):

    def fit_one(function, ycol):

        best_rmsle = None

        best_params = None

        # best_cov = None

        # best_y_hat = None

        for i in range(n_samples):

            params, cov, rmsle, y_hat = fit(

                function,

                df["DayOfYear"].to_numpy(),

                df[ycol].to_numpy(),

                maxfev=maxfev,

            )

            if rmsle >= (best_rmsle or rmsle):

                best_rmsle = rmsle

                best_params = params

        result = {f"{ycol}_rmsle": best_rmsle}

        result.update({f"{ycol}_p_{i}": p for i, p in enumerate(best_params)})

        return result



    result = {}

    result.update(fit_one(logistic, "ConfirmedCases"))

    result.update(fit_one(logistic, "Fatalities"))

    return pd.DataFrame([result])
train_fit = train.groupby(

    ["Country_Region", "Province_State"], observed=True, sort=False

).apply(lambda x: fit_model(x, n_samples=20, maxfev=1000))
print(train_fit.head())

print(train_fit.describe())
def predict(df):

    def predict_one(col):

        df[f"predict_{col}"] = logistic(

            df["DayOfYear"].to_numpy(),

            df[f"{col}_p_0"].to_numpy(),

            df[f"{col}_p_1"].to_numpy(),

            df[f"{col}_p_2"].to_numpy(),

        )



    predict_one("ConfirmedCases")

    predict_one("Fatalities")
test = pd.merge(

    test,

    train_fit.reset_index(),

    on=["Country_Region", "Province_State"],

    how="left",

)

predict(test)
print(test.head())

print(test.describe())
submission = test[["ForecastId", "predict_ConfirmedCases", "predict_Fatalities"]].rename(

    columns={"predict_ConfirmedCases": "ConfirmedCases", "predict_Fatalities": "Fatalities"})

submission[["ConfirmedCases", "Fatalities"]] = submission[["ConfirmedCases", "Fatalities"]].round().astype(int)

submission.to_csv('submission.csv', index=False)

print(submission.head())

print(submission.describe())