!pip install ../input/stochastic/stochastic-0.4.0-py2.py3-none-any.whl
import gc



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.metrics import f1_score

from sklearn.model_selection import KFold



import lightgbm as lgb
SEED = 42

N_SPLITS = 5
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
def macro_f1_score(preds, train_data):

    labels = train_data.get_label()

    preds = np.round(np.clip(preds, 0, 10)).astype(int)

    score = f1_score(labels, preds, average="macro")

    return "f1", score, True
def features(df):

    df = df.sort_values(by=["time"]).reset_index(drop=True)

    df.index = df.time * 10000 - 1

    df["batch"] = df.index // 25000

    df["batch_index"] = df.index - df.batch * 25000

    df["batch_slices"] = df["batch_index"] // 2500

    df["batch_slices_2"] = df["batch"].astype(str) + "_" + df["batch_slices"].astype(str)



    for c in ["batch", "batch_slices_2"]:

        d = {}



        d["mean_" + c] = df.groupby([c])["signal"].mean()

        d["std_" + c] = df.groupby([c])["signal"].std()

        d["median_" + c] = df.groupby([c])["signal"].median()

        d["min_" + c] = df.groupby([c])["signal"].min()

        d["max_" + c] = df.groupby([c])["signal"].max()

        d["mean_abs_change_" + c] = df.groupby([c])["signal"].apply(lambda x: np.mean(np.abs(np.diff(x))))

        d["abs_max_" + c] = df.groupby([c])["signal"].apply(lambda x: np.max(np.abs(x)))

        d["abs_min_" + c] = df.groupby([c])["signal"].apply(lambda x: np.min(np.abs(x)))

        d["range_" + c] = d["max_" + c] - d["min_" + c]

        d["abs_range_" + c] = np.abs(d["max_" + c] - d["min_" + c])

        d["max_to_min_" + c] = d["max_" + c] / d["min_" + c]

        d["abs_avg_" + c] = (d["abs_min_" + c] + d["abs_max_" + c]) / 2



        for v in d:

            df[v] = df[c].map(d[v].to_dict())



    df["signal_shift_+1"] = [0] + list(df["signal"].values[:-1])

    df["signal_shift_-1"] = list(df["signal"].values[1:]) + [0]

    for i in df[df["batch_index"] == 0].index:

        df["signal_shift_+1"][i] = np.nan

    for i in df[df["batch_index"] == 49999].index:

        df["signal_shift_-1"][i] = np.nan



    for c in [x for x in df.columns if x not in ["time", "signal", "open_channels", "batch", "batch_index", "batch_slices", "batch_slices_2"]]:

        df[c + "_minus_signal"] = df[c] - df["signal"]



    return df
train = pd.read_csv("/kaggle/input/liverpool-ion-switching/train.csv")

test = pd.read_csv("/kaggle/input/liverpool-ion-switching/test.csv")
train["signal"] = np.exp(train["signal"])

test["signal"] = np.exp(test["signal"])
%%time

train = features(train)

test = features(test)
%%time

train = reduce_memory_usage(train)

test = reduce_memory_usage(test)
features = [x for x in train.columns if x not in ["time", "open_channels", "batch", "batch_index", "batch_slices", "batch_slices_2"]]

target = "open_channels"
X, y = train[features], train[target]
del train

gc.collect()
params = {

    "learning_rate": 0.1,

    "max_depth": -1,

    "num_leaves": 2**7 + 1,

    "metric": "l2",

    "random_state": SEED,

    "n_jobs": -1,

    "sample_fraction": 0.33

}
folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

preds = np.zeros(len(test))
feature_importance = pd.DataFrame()

feature_importance["Feature"] = X.columns

feature_importance["Value"] = 0
for fold, (train_idx, test_idx) in enumerate(folds.split(X)):

    print("Fold", fold)



    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]

    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]



    model = lgb.train(params, train_set=lgb.Dataset(X_train, y_train), num_boost_round=2000, valid_sets=lgb.Dataset(X_test, y_test), feval=macro_f1_score, early_stopping_rounds=100, verbose_eval=100)

    preds += model.predict(test[features], num_iteration=model.best_iteration)



    current_importance = pd.DataFrame(zip(X.columns, model.feature_importance()), columns=["Feature", "Value"])

    feature_importance = pd.concat((feature_importance, current_importance)).groupby("Feature", as_index=False).sum()
preds = preds / N_SPLITS
test["open_channels"] = np.round(np.clip(preds, 0, 10)).astype(int)

test[["time", "open_channels"]].to_csv("submission.csv", index=False, float_format="%.4f")
fig = plt.figure(figsize=(20, 40))

fig.patch.set_facecolor("white")

sns.set(style="whitegrid")

sns.barplot(x="Value", y="Feature", data=feature_importance.sort_values(by="Value", ascending=False))

plt.title("LightGBM feature importance")

plt.tight_layout()

plt.show()
plt.savefig("feature_importance.png")
feature_importance.sort_values("Value", ascending=False).to_csv("feature_importance.csv", index=False)