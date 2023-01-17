import os

import gc

import warnings



import pandas as pd

from pandas.plotting import register_matplotlib_converters

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder



warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 500)

pd.set_option("display.max_rows", 500)

register_matplotlib_converters()

sns.set()
import IPython





def display(*dfs, head=True):

    for df in dfs:

        IPython.display.display(df.head() if head else df)
def on_kaggle():

    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ
if on_kaggle():

    os.system("pip install --quiet mlflow_extend")
def reduce_mem_usage(df, verbose=False):

    start_mem = df.memory_usage().sum() / 1024 ** 2

    int_columns = df.select_dtypes(include=["int"]).columns

    float_columns = df.select_dtypes(include=["float"]).columns



    for col in int_columns:

        df[col] = pd.to_numeric(df[col], downcast="integer")



    for col in float_columns:

        df[col] = pd.to_numeric(df[col], downcast="float")



    end_mem = df.memory_usage().sum() / 1024 ** 2

    if verbose:

        print(

            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(

                end_mem, 100 * (start_mem - end_mem) / start_mem

            )

        )

    return df
def read_data():

    INPUT_DIR = "/kaggle/input" if on_kaggle() else "input"

    INPUT_DIR = f"{INPUT_DIR}/m5-forecasting-accuracy"



    print("Reading files...")



    calendar = pd.read_csv(f"{INPUT_DIR}/calendar.csv").pipe(reduce_mem_usage)

    prices = pd.read_csv(f"{INPUT_DIR}/sell_prices.csv").pipe(reduce_mem_usage)



    sales = pd.read_csv(f"{INPUT_DIR}/sales_train_validation.csv",).pipe(

        reduce_mem_usage

    )

    submission = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv").pipe(

        reduce_mem_usage

    )



    print("sales shape:", sales.shape)

    print("prices shape:", prices.shape)

    print("calendar shape:", calendar.shape)

    print("submission shape:", submission.shape)



    # calendar shape: (1969, 14)

    # sell_prices shape: (6841121, 4)

    # sales_train_val shape: (30490, 1919)

    # submission shape: (60980, 29)



    return sales, prices, calendar, submission
sales, prices, calendar, submission = read_data()



NUM_ITEMS = sales.shape[0]  # 30490

DAYS_PRED = submission.shape[1] - 1  # 28
def encode_categorical(df, cols):

    for col in cols:

        # Leave NaN as it is.

        le = LabelEncoder()

        not_null = df[col][df[col].notnull()]

        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)



    return df





calendar = encode_categorical(

    calendar, ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]

).pipe(reduce_mem_usage)



sales = encode_categorical(

    sales, ["item_id", "dept_id", "cat_id", "store_id", "state_id"],

).pipe(reduce_mem_usage)



prices = encode_categorical(prices, ["item_id", "store_id"]).pipe(reduce_mem_usage)
def extract_num(ser):

    return ser.str.extract(r"(\d+)").astype(np.int16)





def reshape_sales(sales, submission, d_thresh=0, verbose=True):

    # melt sales data, get it ready for training

    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]



    # get product table.

    product = sales[id_columns]



    sales = sales.melt(id_vars=id_columns, var_name="d", value_name="demand",)

    sales = reduce_mem_usage(sales)



    # separate test dataframes.

    vals = submission[submission["id"].str.endswith("validation")]

    evals = submission[submission["id"].str.endswith("evaluation")]



    # change column names.

    vals.columns = ["id"] + [f"d_{d}" for d in range(1914, 1914 + DAYS_PRED)]

    evals.columns = ["id"] + [f"d_{d}" for d in range(1942, 1942 + DAYS_PRED)]



    # merge with product table

    evals["id"] = evals["id"].str.replace("_evaluation", "_validation")

    vals = vals.merge(product, how="left", on="id")

    evals = evals.merge(product, how="left", on="id")

    evals["id"] = evals["id"].str.replace("_validation", "_evaluation")



    if verbose:

        print("validation")

        display(vals)



        print("evaluation")

        display(evals)



    vals = vals.melt(id_vars=id_columns, var_name="d", value_name="demand")

    evals = evals.melt(id_vars=id_columns, var_name="d", value_name="demand")



    sales["part"] = "train"

    vals["part"] = "validation"

    evals["part"] = "evaluation"



    data = pd.concat([sales, vals, evals], axis=0)



    del sales, vals, evals



    data["d"] = extract_num(data["d"])

    data = data[data["d"] >= d_thresh]



    # delete evaluation for now.

    data = data[data["part"] != "evaluation"]



    gc.collect()



    if verbose:

        print("data")

        display(data)



    return data





def merge_calendar(data, calendar):

    calendar = calendar.drop(["weekday", "wday", "month", "year"], axis=1)

    return data.merge(calendar, how="left", on="d")





def merge_prices(data, prices):

    return data.merge(prices, how="left", on=["store_id", "item_id", "wm_yr_wk"])
data = reshape_sales(sales, submission, d_thresh=1941 - int(365 * 2))

del sales

gc.collect()



calendar["d"] = extract_num(calendar["d"])

data = merge_calendar(data, calendar)

del calendar

gc.collect()



data = merge_prices(data, prices)

del prices

gc.collect()



data = reduce_mem_usage(data)
def add_demand_features(df):

    for diff in [0, 1, 2]:

        shift = DAYS_PRED + diff

        df[f"shift_t{shift}"] = df.groupby(["id"])["demand"].transform(

            lambda x: x.shift(shift)

        )



    for window in [7, 30, 60, 90, 180]:

        df[f"rolling_std_t{window}"] = df.groupby(["id"])["demand"].transform(

            lambda x: x.shift(DAYS_PRED).rolling(window).std()

        )



    for window in [7, 30, 60, 90, 180]:

        df[f"rolling_mean_t{window}"] = df.groupby(["id"])["demand"].transform(

            lambda x: x.shift(DAYS_PRED).rolling(window).mean()

        )



    for window in [7, 30, 60]:

        df[f"rolling_min_t{window}"] = df.groupby(["id"])["demand"].transform(

            lambda x: x.shift(DAYS_PRED).rolling(window).min()

        )



    for window in [7, 30, 60]:

        df[f"rolling_max_t{window}"] = df.groupby(["id"])["demand"].transform(

            lambda x: x.shift(DAYS_PRED).rolling(window).max()

        )



    df["rolling_skew_t30"] = df.groupby(["id"])["demand"].transform(

        lambda x: x.shift(DAYS_PRED).rolling(30).skew()

    )

    df["rolling_kurt_t30"] = df.groupby(["id"])["demand"].transform(

        lambda x: x.shift(DAYS_PRED).rolling(30).kurt()

    )

    return df





def add_price_features(df):

    df["shift_price_t1"] = df.groupby(["id"])["sell_price"].transform(

        lambda x: x.shift(1)

    )

    df["price_change_t1"] = (df["shift_price_t1"] - df["sell_price"]) / (

        df["shift_price_t1"]

    )

    df["rolling_price_max_t365"] = df.groupby(["id"])["sell_price"].transform(

        lambda x: x.shift(1).rolling(365).max()

    )

    df["price_change_t365"] = (df["rolling_price_max_t365"] - df["sell_price"]) / (

        df["rolling_price_max_t365"]

    )



    df["rolling_price_std_t7"] = df.groupby(["id"])["sell_price"].transform(

        lambda x: x.rolling(7).std()

    )

    df["rolling_price_std_t30"] = df.groupby(["id"])["sell_price"].transform(

        lambda x: x.rolling(30).std()

    )

    return df.drop(["rolling_price_max_t365", "shift_price_t1"], axis=1)





def add_time_features(df, dt_col):

    df[dt_col] = pd.to_datetime(df[dt_col])

    attrs = [

        "year",

        "quarter",

        "month",

        "week",

        "day",

        "dayofweek",

    ]



    for attr in attrs:

        dtype = np.int16 if attr == "year" else np.int8

        df[attr] = getattr(df[dt_col].dt, attr).astype(dtype)



    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(np.int8)

    return df
data = add_demand_features(data).pipe(reduce_mem_usage)

data = add_price_features(data).pipe(reduce_mem_usage)

dt_col = "date"

data = add_time_features(data, dt_col).pipe(reduce_mem_usage)

data = data.sort_values("date")



print("start date:", data[dt_col].min())

print("end date:", data[dt_col].max())

print("data shape:", data.shape)
class CustomTimeSeriesSplitter:

    def __init__(self, n_splits=5, train_days=80, test_days=20, day_col="d"):

        self.n_splits = n_splits

        self.train_days = train_days

        self.test_days = test_days

        self.day_col = day_col



    def split(self, X, y=None, groups=None):

        SEC_IN_DAY = 3600 * 24

        sec = (X[self.day_col] - X[self.day_col].iloc[0]) * SEC_IN_DAY

        duration = sec.max()



        train_sec = self.train_days * SEC_IN_DAY

        test_sec = self.test_days * SEC_IN_DAY

        total_sec = test_sec + train_sec



        if self.n_splits == 1:

            train_start = duration - total_sec

            train_end = train_start + train_sec



            train_mask = (sec >= train_start) & (sec < train_end)

            test_mask = sec >= train_end



            yield sec[train_mask].index.values, sec[test_mask].index.values



        else:

            # step = (duration - total_sec) / (self.n_splits - 1)

            step = DAYS_PRED * SEC_IN_DAY



            for idx in range(self.n_splits):

                # train_start = idx * step

                shift = (self.n_splits - (idx + 1)) * step

                train_start = duration - total_sec - shift

                train_end = train_start + train_sec

                test_end = train_end + test_sec



                train_mask = (sec > train_start) & (sec <= train_end)



                if idx == self.n_splits - 1:

                    test_mask = sec > train_end

                else:

                    test_mask = (sec > train_end) & (sec <= test_end)



                yield sec[train_mask].index.values, sec[test_mask].index.values



    def get_n_splits(self):

        return self.n_splits
day_col = "d"

cv_params = {

    "n_splits": 3,

    "train_days": int(365 * 1.5),

    "test_days": DAYS_PRED,

    "day_col": day_col,

}

cv = CustomTimeSeriesSplitter(**cv_params)
def show_cv_days(cv, X, dt_col, day_col):

    for ii, (tr, tt) in enumerate(cv.split(X)):

        print(f"----- Fold: ({ii + 1} / {cv.n_splits}) -----")

        tr_start = X.iloc[tr][dt_col].min()

        tr_end = X.iloc[tr][dt_col].max()

        tr_days = X.iloc[tr][day_col].max() - X.iloc[tr][day_col].min() + 1



        tt_start = X.iloc[tt][dt_col].min()

        tt_end = X.iloc[tt][dt_col].max()

        tt_days = X.iloc[tt][day_col].max() - X.iloc[tt][day_col].min() + 1



        df = pd.DataFrame(

            {

                "start": [tr_start, tt_start],

                "end": [tr_end, tt_end],

                "days": [tr_days, tt_days],

            },

            index=["train", "test"],

        )



        display(df)





def plot_cv_indices(cv, X, dt_col, lw=10):

    n_splits = cv.get_n_splits()

    _, ax = plt.subplots(figsize=(20, n_splits))



    # Generate the training/testing visualizations for each CV split

    for ii, (tr, tt) in enumerate(cv.split(X)):

        # Fill in indices with the training/test groups

        indices = np.array([np.nan] * len(X))

        indices[tt] = 1

        indices[tr] = 0



        # Visualize the results

        ax.scatter(

            X[dt_col],

            [ii + 0.5] * len(indices),

            c=indices,

            marker="_",

            lw=lw,

            cmap=plt.cm.coolwarm,

            vmin=-0.2,

            vmax=1.2,

        )



    # Formatting

    MIDDLE = 15

    LARGE = 20

    ax.set_xlabel("Datetime", fontsize=LARGE)

    ax.set_xlim([X[dt_col].min(), X[dt_col].max()])

    ax.set_ylabel("CV iteration", fontsize=LARGE)

    ax.set_yticks(np.arange(n_splits) + 0.5)

    ax.set_yticklabels(list(range(n_splits)))

    ax.invert_yaxis()

    ax.tick_params(axis="both", which="major", labelsize=MIDDLE)

    ax.set_title("{}".format(type(cv).__name__), fontsize=LARGE)

    return ax
sample = data.iloc[::1000][[day_col, dt_col]].reset_index(drop=True)

show_cv_days(cv, sample, dt_col, day_col)

plot_cv_indices(cv, sample, dt_col)



del sample

gc.collect()
features = [

    "item_id",

    "dept_id",

    "cat_id",

    "store_id",

    "state_id",

    "event_name_1",

    "event_type_1",

    "event_name_2",

    "event_type_2",

    "snap_CA",

    "snap_TX",

    "snap_WI",

    "sell_price",

    # demand features

    "shift_t28",

    "shift_t29",

    "shift_t30",

    # std

    "rolling_std_t7",

    "rolling_std_t30",

    "rolling_std_t60",

    "rolling_std_t90",

    "rolling_std_t180",

    # mean

    "rolling_mean_t7",

    "rolling_mean_t30",

    "rolling_mean_t60",

    "rolling_mean_t90",

    "rolling_mean_t180",

    # min

    "rolling_min_t7",

    "rolling_min_t30",

    "rolling_min_t60",

    # max

    "rolling_max_t7",

    "rolling_max_t30",

    "rolling_max_t60",

    # others

    "rolling_skew_t30",

    "rolling_kurt_t30",

    # price features

    "price_change_t1",

    "price_change_t365",

    "rolling_price_std_t7",

    "rolling_price_std_t30",

    # time features

    "year",

    "quarter",

    "month",

    "week",

    "day",

    "dayofweek",

    "is_weekend",

]



# prepare training and test data.

# 2011-01-29 ~ 2016-04-24 : d_1    ~ d_1913

# 2016-04-25 ~ 2016-05-22 : d_1914 ~ d_1941 (public)

# 2016-05-23 ~ 2016-06-19 : d_1942 ~ d_1969 (private)



is_train = data["d"] < 1914



# Attach "d" to X_train for cross validation.

X_train = data[is_train][[day_col] + features].reset_index(drop=True)

y_train = data[is_train]["demand"].reset_index(drop=True)

X_test = data[~is_train][features].reset_index(drop=True)



# keep these two columns to use later.

id_date = data[~is_train][["id", "date"]].reset_index(drop=True)



del data

gc.collect()



print("X_train shape:", X_train.shape)

print("X_test shape:", X_test.shape)
def train_lgb(bst_params, fit_params, X, y, cv, drop_when_train=None):

    models = []



    if drop_when_train is None:

        drop_when_train = []



    for idx_fold, (idx_trn, idx_val) in enumerate(cv.split(X, y)):

        print(f"\n----- Fold: ({idx_fold + 1} / {cv.get_n_splits()}) -----\n")



        X_trn, X_val = X.iloc[idx_trn], X.iloc[idx_val]

        y_trn, y_val = y.iloc[idx_trn], y.iloc[idx_val]

        train_set = lgb.Dataset(

            X_trn.drop(drop_when_train, axis=1),

            label=y_trn,

            categorical_feature=["item_id"],

        )

        val_set = lgb.Dataset(

            X_val.drop(drop_when_train, axis=1),

            label=y_val,

            categorical_feature=["item_id"],

        )



        model = lgb.train(

            bst_params,

            train_set,

            valid_sets=[train_set, val_set],

            valid_names=["train", "valid"],

            **fit_params,

        )

        models.append(model)



        del idx_trn, idx_val, X_trn, X_val, y_trn, y_val

        gc.collect()



    return models
bst_params = {

    "boosting_type": "gbdt",

    "metric": "rmse",

    "objective": "regression",

    "n_jobs": -1,

    "seed": 42,

    "learning_rate": 0.1,

    "bagging_fraction": 0.75,

    "bagging_freq": 10,

    "colsample_bytree": 0.75,

}



fit_params = {

    "num_boost_round": 100_000,

    "early_stopping_rounds": 50,

    "verbose_eval": 100,

}



models = train_lgb(

    bst_params, fit_params, X_train, y_train, cv, drop_when_train=[day_col]

)



del X_train, y_train

gc.collect()
def rmse(y_true, y_pred):

    return np.sqrt(mean_squared_error(y_true, y_pred))
imp_type = "gain"

importances = np.zeros(X_test.shape[1])

preds = np.zeros(X_test.shape[0])



for model in models:

    preds += model.predict(X_test)

    importances += model.feature_importance(imp_type)



preds = preds / cv.get_n_splits()

importances = importances / cv.get_n_splits()
from mlflow_extend import mlflow, plotting as mplt



with mlflow.start_run():

    mlflow.log_params_flatten({"bst": bst_params, "fit": fit_params, "cv": cv_params})





features = models[0].feature_name()

_ = mplt.feature_importance(features, importances, imp_type, limit=30)
def make_submission(test, submission):

    preds = test[["id", "date", "demand"]]

    preds = preds.pivot(index="id", columns="date", values="demand").reset_index()

    preds.columns = ["id"] + ["F" + str(d + 1) for d in range(DAYS_PRED)]



    vals = submission[["id"]].merge(preds, how="inner", on="id")

    evals = submission[submission["id"].str.endswith("evaluation")]

    final = pd.concat([vals, evals])



    assert final.drop("id", axis=1).isnull().sum().sum() == 0

    assert final["id"].equals(submission["id"])



    final.to_csv("submission.csv", index=False)
make_submission(id_date.assign(demand=preds), submission)