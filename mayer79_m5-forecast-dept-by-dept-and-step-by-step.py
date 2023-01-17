import pandas as pd

import numpy as np

import os

import lightgbm as lgb
path = "../input/m5-forecasting-accuracy"



calendar = pd.read_csv(os.path.join(path, "calendar.csv"))

selling_prices = pd.read_csv(os.path.join(path, "sell_prices.csv"))

sample_submission = pd.read_csv(os.path.join(path, "sample_submission.csv"))

sales = pd.read_csv(os.path.join(path, "sales_train_validation.csv"))
from sklearn.preprocessing import OrdinalEncoder



def prep_calendar(df):

    df = df.drop(["date", "weekday", "event_type_1", "event_type_2"], axis=1)

    df = df.assign(d = df.d.str[2:].astype(int))

    to_ordinal = ["event_name_1", "event_name_2"] 

    df[to_ordinal] = df[to_ordinal].fillna("1")

    df[to_ordinal] = OrdinalEncoder(dtype="int").fit_transform(df[to_ordinal]) + 1

    to_int8 = ["wday", "month", "snap_CA", "snap_TX", "snap_WI"] + to_ordinal

    df[to_int8] = df[to_int8].astype("int8")

    

    return df



calendar = prep_calendar(calendar)

calendar.head()
from sklearn.model_selection import train_test_split



LAGS = [7, 28]

WINDOWS = [7, 28, 56]

FIRST = 1914

LENGTH = 28



def demand_features(df):

    """ Derive features from sales data and remove rows with missing values """

    

    for lag in LAGS:

        df[f'lag_t{lag}'] = df.groupby('id')['demand'].transform(lambda x: x.shift(lag)).astype("float32")

        for w in WINDOWS:

            df[f'rolling_mean_lag{lag}_w{w}'] = df.groupby('id')[f'lag_t{lag}'].transform(lambda x: x.rolling(w).mean()).astype("float32")

        

    return df



def demand_features_eval(df):

    """ Same as demand_features but for the step-by-step evaluation """

    out = df.groupby('id', sort=False).last()

    for lag in LAGS:

        out[f'lag_t{lag}'] = df.groupby('id', sort=False)['demand'].nth(-lag-1).astype("float32")

        for w in WINDOWS:

            out[f'rolling_mean_lag{lag}_w{w}'] = df.groupby('id', sort=False)['demand'].nth(list(range(-lag-w, -lag))).groupby('id', sort=False).mean().astype("float32")

    

    return out.reset_index()



def prep_data(df, drop_d=1000, dept_id="FOODS_1"):

    """ Prepare model data sets """

    

    print(f"\nWorking on dept {dept_id}")

    # Filter on dept_id

    df = df[df.dept_id == dept_id]

    df = df.drop(["dept_id", "cat_id"], axis=1)

    

    # Kick out old dates

    df = df.drop(["d_" + str(i+1) for i in range(drop_d)], axis=1)



    # Reshape to long

    df = df.assign(id=df.id.str.replace("_validation", ""))

    df = df.reindex(columns=df.columns.tolist() + ["d_" + str(FIRST + i) for i in range(2 * LENGTH)])

    df = df.melt(id_vars=["id", "item_id", "store_id", "state_id"], var_name='d', value_name='demand')

    df = df.assign(d=df.d.str[2:].astype("int64"),

                   demand=df.demand.astype("float32"))

    

    # Add demand features

    df = demand_features(df)

    

    # Remove rows with NAs

    df = df[df.d > (drop_d + max(LAGS) + max(WINDOWS))]

 

    # Join calendar & prices

    df = df.merge(calendar, how="left", on="d")

    df = df.merge(selling_prices, how="left", on=["store_id", "item_id", "wm_yr_wk"])

    df = df.drop(["wm_yr_wk"], axis=1)

    

    # Ordinal encoding of remaining categorical fields

    for v in ["item_id", "store_id", "state_id"]:

        df[v] = OrdinalEncoder(dtype="int").fit_transform(df[[v]]).astype("int16") + 1

    

    # Determine list of covariables

    x = list(set(df.columns) - {'id', 'd', 'demand'})

            

    # Split into test, valid, train

    test = df[df.d >= FIRST - max(LAGS) - max(WINDOWS)]

    df = df[df.d < FIRST]



    xtrain, xvalid, ytrain, yvalid = train_test_split(df[x], df["demand"], test_size=0.1, shuffle=True, random_state=54)

    train = lgb.Dataset(xtrain, label = ytrain)

    valid = lgb.Dataset(xvalid, label = yvalid)



    return train, valid, test, x



def fit_model(train, valid, dept):

    """ Fit LightGBM model """

     

    params = {

        'metric': 'rmse',

        'objective': 'poisson',

        'seed': 200,

        'learning_rate': 0.2 - 0.13 * (dept in ["HOBBIES_1", "HOBBIES_2", "HOUSEHOLD_2"]),

        'lambda': 0.1,

        'num_leaves': 50,

        'colsample_bytree': 0.7

    }



    fit = lgb.train(params, 

                    train, 

                    num_boost_round = 5000, 

                    valid_sets = [valid], 

                    early_stopping_rounds = 200,

                    verbose_eval = 200)

    

    lgb.plot_importance(fit, importance_type="gain", precision=0, height=0.5, figsize=(6, 10), title=dept);

    

    return fit



def pred_to_csv(fit, test, x, cols=sample_submission.columns, file="submission.csv", first=False):

    """ Calculate predictions and append to submission csv """

    

    # Recursive prediction

    for i, day in enumerate(np.arange(FIRST, FIRST + LENGTH)):

        test_day = demand_features_eval(test[(test.d <= day) & (test.d >= day - max(LAGS) - max(WINDOWS))])

        test.loc[test.d == day, "demand"] = fit.predict(test_day[x]) * 1.03 # https://www.kaggle.com/kyakovlev/m5-dark-magic

    

    # Prepare for reshaping

    test = test.assign(id=test.id + "_" + np.where(test.d < FIRST + LENGTH, "validation", "evaluation"),

                       F="F" + (test.d - FIRST + 1 - LENGTH * (test.d >= FIRST + LENGTH)).astype("str"))

    

    # Reshape

    submission = test.pivot(index="id", columns="F", values="demand").reset_index()[cols].fillna(1)

    

    # Export

    submission.to_csv(file, index=False, mode='w' if first else 'a', header=first)

    

    return True
for i, dept in enumerate(np.unique(sales.dept_id)):

    train, valid, test, x = prep_data(sales, 1150, dept)

    fit = fit_model(train, valid, dept)

    pred_to_csv(fit, test, x, first=(i==0))