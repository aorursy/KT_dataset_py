import datetime

import gc

import math

import os

import random

import sys

import warnings

import itertools



import lightgbm as lgb

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from scipy.stats import randint, uniform

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from sklearn.preprocessing import LabelEncoder



import seaborn as sns

from tqdm import tqdm_notebook as tqdm



warnings.filterwarnings("ignore")
def seed_everything(seed=0):

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)
SEED = 1337

seed_everything(SEED)
def reduce_memory_usage(df):

    numerics = ["int16", "int32", "int64", "float64"]

    for col, col_type in df.dtypes.iteritems():

        best_type = None

        if col_type == "object":

            df[col] = df[col].astype("category")

            best_type = "category"

        elif col_type in numerics:

            downcast = "integer" if "int" in str(col_type) else "float"

            df[col] = pd.to_numeric(df[col], downcast=downcast)

            best_type = df[col].dtype.name

    return df
%%time

train = pd.read_pickle("../input/ieeecis-fraud-detection-merged/train.pkl")

test = pd.read_pickle("../input/ieeecis-fraud-detection-merged/test.pkl")
def get_device_manufacturer(device_info):

    device_info = str(device_info).lower()

    

    if any(s in device_info for s in ["windows", "microsoft", "win64", "wow64"]):

        return "Microsoft"

    

    if any(s in device_info for s in ["ios", "macos", "iphone"]):

        return "Apple"

    

    if any(s in device_info for s in ["pixel"]):

        return "Google"

    

    if any(s in device_info for s in ["samsung", "sm-", "gt-"]):

        return "Samsung"

    

    if any(s in device_info for s in ["redmi", "mi"]):

        return "Xiaomi"

    

    if any(s in device_info for s in ["huawei", "ale-"]):

        return "Huawei"

    

    if any(s in device_info for s in ["lg", "nexus", "lm-", "vs"]):

        return "LG"

    

    if any(s in device_info for s in ["htc", "0paj5", "0pja2", "0pm92", "2pq93", "2ps64", "2pyb2", "2pzc5"]):

        return "HTC"

    

    if any(s in device_info for s in ["moto", "xt"]):

        return "Motorola"

    

    if any(s in device_info for s in ["zte", "blade", "z970"]):

        return "ZTE"

    

    if any(s in device_info for s in ["lenovo"]):

        return "Lenovo"

    

    if any(s in device_info for s in ["asus"]):

        return "ASUS"

    

    if any(s in device_info for s in ["lt22i"]):

        return "Sony"

    

    if any(s in device_info for s in ["nokia", "ta-"]):

        return "Nokia"

    

    if any(s in device_info for s in ["oneplus"]):

        return "OnePlus"

    

    if any(s in device_info for s in ["verykool"]):

        return "Verykool"

    

    if any(s in device_info for s in ["linux"]):

        return "Linux"

    

    if any(s in device_info for s in ["rv"]):

        return "RV"

    

    if any(s in device_info for s in ["m4"]):

        return "M4TEL"

    

    if any(s in device_info for s in ["alcatel", "one touch", "4013m"]):

        return "Alcatel"

    

    if any(s in device_info for s in ["ilium"]):

        return "Lanix"

    

    if any(s in device_info for s in ["hisense"]):

        return "Hisense"

    

    if any(s in device_info for s in ["rct"]):

        return "RCA"

    

    if any(s in device_info for s in ["kfa", "kfd", "kff", "kfg", "kfj", "kfk", "kfm", "kfs", "kft"]):

        return "Amazon"

    

    return "Other"
%%time

for df in [train, test]:

    df["day"] = np.floor((df["TransactionDT"] / (3600 * 24) - 1) % 7).astype(int)

    df["hour"] = (np.floor(df["TransactionDT"] / 3600) % 24).astype(int)

    

    df["TransactionAmt_log"] = np.log(df["TransactionAmt"])

    df["TransactionAmt_int"] = df["TransactionAmt"].astype(int)

    df["TransactionAmt_dec"] = (1000 * (df["TransactionAmt"] - df["TransactionAmt_int"])).astype(int)

    

    df["card_id"] = df["card1"].astype(str) + "__" + df["card2"].astype(str) + "__" + df["card3"].astype(str) + "__" + df["card4"].astype(str) + "__" + df["card5"].astype(str)

        

    for a, b in itertools.combinations(["card1", "card2", "card3", "card4", "card5", "addr1", "addr2", "dist1", "dist2"], 2):

        df["{}__{}".format(a, b)] = df[a].astype(str) + "__" + df[b].astype(str)

    

    df[["P_emaildomain_0", "P_emaildomain_1", "P_emaildomain_2"]] = df["P_emaildomain"].str.split(".", expand=True)

    df[["R_emaildomain_0", "R_emaildomain_1", "R_emaildomain_2"]] = df["R_emaildomain"].str.split(".", expand=True)

    df[["operating_system_0", "operating_system_1", "operating_system_2", "operating_system_3"]] = df["id_30"].str.split(" ", expand=True)

    df[["browser_0", "browser_1", "browser_2", "browser_3"]] = df["id_31"].str.split(" ", expand=True)

    df[["resolution_width", "resolution_height"]] = df["id_33"].str.split("x", expand=True)

    df[["DeviceInfo_0", "DeviceInfo_1", "DeviceInfo_2", "DeviceInfo_3", "DeviceInfo_4"]] = df["DeviceInfo"].str.split(r"[ -/_]", expand=True)[[0, 1, 2, 3, 4]]



    df["TransactionAmt_to_mean_card1"] = df["TransactionAmt"] / df.groupby(["card1"])["TransactionAmt"].transform("mean")

    df["TransactionAmt_to_mean_card4"] = df["TransactionAmt"] / df.groupby(["card4"])["TransactionAmt"].transform("mean")

    df["TransactionAmt_to_std_card1"] = df["TransactionAmt"] / df.groupby(["card1"])["TransactionAmt"].transform("std")

    df["TransactionAmt_to_std_card4"] = df["TransactionAmt"] / df.groupby(["card4"])["TransactionAmt"].transform("std")



    df["id_02_to_mean_card1"] = df["id_02"] / df.groupby(["card1"])["id_02"].transform("mean")

    df["id_02_to_mean_card4"] = df["id_02"] / df.groupby(["card4"])["id_02"].transform("mean")

    df["id_02_to_std_card1"] = df["id_02"] / df.groupby(["card1"])["id_02"].transform("std")

    df["id_02_to_std_card4"] = df["id_02"] / df.groupby(["card4"])["id_02"].transform("std")



    df["D15_to_mean_card1"] = df["D15"] / df.groupby(["card1"])["D15"].transform("mean")

    df["D15_to_mean_card4"] = df["D15"] / df.groupby(["card4"])["D15"].transform("mean")

    df["D15_to_std_card1"] = df["D15"] / df.groupby(["card1"])["D15"].transform("std")

    df["D15_to_std_card4"] = df["D15"] / df.groupby(["card4"])["D15"].transform("std")



    df["D15_to_mean_addr1"] = df["D15"] / df.groupby(["addr1"])["D15"].transform("mean")

    df["D15_to_mean_addr2"] = df["D15"] / df.groupby(["addr2"])["D15"].transform("mean")

    df["D15_to_std_addr1"] = df["D15"] / df.groupby(["addr1"])["D15"].transform("std")

    df["D15_to_std_addr2"] = df["D15"] / df.groupby(["addr2"])["D15"].transform("std")

    

    df["device_manufacturer"] = df["DeviceInfo"].map(lambda x: get_device_manufacturer(x))
default_categorical_columns = ["ProductCD", "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2", "P_emaildomain", "R_emaildomain", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "id_12", "id_13", "id_14", "id_15", "id_16", "id_17", "id_18", "id_19", "id_20", "id_21", "id_22", "id_23", "id_24", "id_25", "id_26", "id_27", "id_28", "id_29", "id_30", "id_31", "id_32", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38", "DeviceType", "DeviceInfo"]
for c in tqdm(set(default_categorical_columns + [c for c in train.columns if train[c].dtype == "object"])):

    labels = {value: index for index, value in enumerate(["__UNKNOWN__"] + list(set(list(train[c].astype(str).values) + list(test[c].astype(str).values))))}

    train[c] = train[c].map(lambda x: labels.get(x)).fillna(labels["__UNKNOWN__"]).astype(int)

    test[c] = test[c].map(lambda x: labels.get(x)).fillna(labels["__UNKNOWN__"]).astype(int)
%%time

train = reduce_memory_usage(train)

test = reduce_memory_usage(test)
lgb_params = {

    "boosting_type": "gbdt",

    "colsample_bytree": 0.5,

    "early_stopping_rounds": 100,

    "learning_rate": 0.005,

    "max_bin": 255,

    "max_depth": -1,

    "metric": "auc",

    "n_estimators": 20000,

    "n_jobs": -1,

    "num_leaves": 2**9,

    "objective": "binary",

    "seed": SEED,

    "subsample": 0.7,

    "subsample_freq": 1,

    "tree_learner": "serial"

}
feature_columns = list(train.columns)

target_column = "isFraud"



columns_to_drop = ["TransactionID", "isFraud", "TransactionDT"]

for c in columns_to_drop:

    feature_columns.remove(c)
print(feature_columns)
X, y = train[feature_columns], train[target_column]

P = test[feature_columns]
%%time

N_SPLITS = 5

folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)



train_preds = np.zeros(train.shape[0])

test_preds = np.zeros(test.shape[0])



del train, test

gc.collect()



feature_importance = pd.DataFrame()

feature_importance["Feature"] = X.columns

feature_importance["Value"] = 0



for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(X, y)):

    print("Fold", fold_idx + 1)

    

    train_x, train_y = X.iloc[train_idx, :], y[train_idx]

    valid_x, valid_y = X.iloc[valid_idx, :], y[valid_idx]

    

    model = lgb.LGBMClassifier(**lgb_params)

    model.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], verbose=100)

    

    train_preds[valid_idx] = model.predict_proba(valid_x, num_iteration=model.best_iteration_)[:, 1]

    test_preds += model.predict_proba(P, num_iteration=model.best_iteration_)[:, 1] / folds.n_splits

    

    print("Fold {} AUC: {:.6f}".format(fold_idx + 1, roc_auc_score(valid_y, train_preds[valid_idx])))

    

    current_importance = pd.DataFrame(zip(X.columns, model.feature_importances_), columns=["Feature", "Value"])

    feature_importance = pd.concat((feature_importance, current_importance)).groupby("Feature", as_index=False).sum()



print("Global AUC: {:.6f}".format(roc_auc_score(y, train_preds)))
feature_importance["Value"] *= 100 / feature_importance["Value"].sum()



fig = plt.figure(figsize=(20, 200))

fig.patch.set_facecolor("white")

sns.set(style="whitegrid")

sns.barplot(x="Value", y="Feature", data=feature_importance.sort_values(by="Value", ascending=False))

plt.title("LightGBM feature importance (%)")

plt.tight_layout()

plt.show()
submission = pd.read_csv("../input/ieee-fraud-detection/sample_submission.csv")

submission[target_column] = test_preds

submission.to_csv("submission.csv", index=False)
plt.savefig("feature_importance.png")
feature_importance.sort_values("Value", ascending=False).to_csv("feature_importance.csv", index=False)
feature_importance.sort_values("Value", ascending=False)[["Feature"]].to_csv("feature_importance_name_only.csv", index=False)