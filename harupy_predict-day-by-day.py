from functools import reduce

import gc

import warnings



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb



from tqdm.notebook import tqdm

import IPython



warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 500)

pd.set_option("display.max_rows", 500)

sns.set()
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





def display(*dfs):

    for df in dfs:

        IPython.display.display(df)
INPUT_DIR = "../input/m5-forecasting-accuracy"



TARGET = "sales"

TRAIN_DAYS = 30

PRED_START = 1914

PRED_DAYS = 28

MAX_LAG = 5
def read_sales():

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    d_cols = [f"d_{d}" for d in range(PRED_START - TRAIN_DAYS, PRED_START)]

    return pd.read_csv(f"{INPUT_DIR}/sales_train_validation.csv", usecols=id_cols + d_cols)
sales = read_sales().pipe(reduce_mem_usage)

sbm_sample = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")
# Comment out this cell after verifying the notebook runs without errors.



# Sample 5 items from each department.

# sales = (

#     sales

#     .groupby("dept_id")

#     .head(5)

#     .reset_index(drop=True)

# )



# item_ids = sales["item_id"].tolist()

# ids = sales["id"].tolist()



# mask = sbm_sample["id"].isin(ids) | sbm_sample["id"].str.replace("_evaluation", "_validation").isin(ids)

# sbm_sample = sbm_sample[sbm_mask].reset_index(drop=True)
def add_pred_cols(df):

    return df.assign(**{f"d_{d}": np.nan for d in range(PRED_START, PRED_START + 2 * PRED_DAYS)})





def melt_sales(sales):

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    return sales.melt(id_vars=id_cols, var_name="d", value_name=TARGET)





def convert_d_to_int(df):

    return df.assign(d=df["d"].str.extract(r"(\d+)").astype(int))





def apply_funcs(df, funcs):

    return reduce(lambda df, f: f(df), funcs, df)
pp_funcs = [

    add_pred_cols,

    melt_sales,

    convert_d_to_int,

    reduce_mem_usage,

]

sales = apply_funcs(sales, pp_funcs)



sales.head()
def add_lag_features(df):

    for lag in range(1, MAX_LAG + 1):

        df[f"lag_{lag}"] = df.groupby(["id"])[TARGET].transform(

            lambda x: x.shift(lag)

        )

    return df





def add_rolling_features(df):

    for lag in [1]:

        for window in [7, 14]:

            df[f"lag_{lag}_rolling_{window}_mean"] = df.groupby(["id"])[f"lag_{lag}"].transform(

                lambda x: x.rolling(window).mean()

            )



    for lag in [1]:

        for window in [7, 14]:

            df[f"lag_{lag}_rolling_{window}_std"] = df.groupby(["id"])[f"lag_{lag}"].transform(

                lambda x: x.rolling(window).std()

            )



    return df
fe_funcs = [

    add_lag_features,

    add_rolling_features,

    reduce_mem_usage,

]

sales = apply_funcs(sales, fe_funcs)



sales.sort_values(["id", "d"]).head(20)
from sklearn.model_selection import train_test_split



drop_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d"]



split_params = {

    "test_size": 0.1,

    "random_state": 42,

}



X_train, X_valid = train_test_split(sales.drop(drop_cols, axis=1).dropna(), **split_params)



y_train = X_train.pop(TARGET)

y_valid = X_valid.pop(TARGET)
print(X_train.shape)

print(X_valid.shape)
trn_set = lgb.Dataset(X_train, y_train)

val_set = lgb.Dataset(X_valid, y_valid)
bst_params = {

    "objective": "poisson",

    "metric": "rmse",

    "learning_rate": 0.1,

    "random_state": 42,

}



train_params = {

    "num_boost_round": 1000,

    "early_stopping_rounds": 50,

    "verbose_eval": 50,

}





model = lgb.train(

    bst_params,

    trn_set,

    valid_sets=[trn_set, val_set],

    valid_names=["train", "valid"],

    **train_params,

)
del trn_set, val_set

gc.collect()
for d in tqdm(range(PRED_START, PRED_START + 2 * PRED_DAYS)):

    sales_sub = sales[(sales["d"] >= d - TRAIN_DAYS) & (sales["d"] <= d)]

    sales_sub = apply_funcs(sales_sub, fe_funcs)

    sales_sub = sales_sub[sales_sub["d"] == d][model.feature_name()]

    sales.loc[sales["d"] == d, "sales"] = model.predict(sales_sub)
display(

    sales[sales["d"] >= PRED_START].head(),

    sales[sales["d"] >= PRED_START].tail(),

)
def reshape_to_submission(pred):

    cols = ["id", "d", "sales"]

    vals = pred[pred["d"].between(PRED_START, PRED_START + PRED_DAYS - 1)][cols]

    evals = pred[pred["d"] >= PRED_START + PRED_DAYS][cols]



    vals = vals.pivot(index="id", columns="d", values=TARGET).reset_index()

    evals = evals.pivot(index="id", columns="d", values=TARGET).reset_index()



    F_cols = ["id"] + ["F" + str(d + 1) for d in range(PRED_DAYS)]

    vals.columns = F_cols

    evals.columns = F_cols



    return pd.concat([

        vals,

        evals.assign(id=evals["id"].str.replace("_validation", "_evaluation"))

    ])
sbm = reshape_to_submission(sales)

sbm = sbm_sample[["id"]].merge(sbm, on="id", how="inner")  # Match id order to submission sample.

sbm.head()
assert sbm.drop("id", axis=1).notnull().all(axis=None)

assert sbm.columns.equals(sbm_sample.columns)

assert sbm["id"].equals(sbm_sample["id"])
sbm.to_csv("submission.csv", index=False)