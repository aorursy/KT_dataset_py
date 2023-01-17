import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import *
import gc
from sklearn.feature_selection import f_classif
import lightgbm as lgbm
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, uniform, norm
from scipy.stats import randint, poisson
from sklearn.metrics import confusion_matrix, make_scorer

sns.set(style="darkgrid", context="notebook")
rand_seed = 135
np.random.seed(rand_seed)
xsize = 12.0
ysize = 8.0

import os
print(os.listdir("../input"))
def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print("Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
%%time

train_meta_df = pd.read_csv("../input/metadata_train.csv")
train_df = pq.read_pandas("../input/train.parquet").to_pandas()
%%time

train_meta_df = reduce_mem_usage(train_meta_df)
train_df = reduce_mem_usage(train_df)
gc.collect()
train_meta_df.shape
train_meta_df.head(6)
train_df.head()
fig, axes = plt.subplots(nrows=2)
fig.set_size_inches(xsize, 2.0*ysize)

sns.countplot(x="phase", data=train_meta_df, ax=axes[0])

sns.countplot(x="target", data=train_meta_df, ax=axes[1])

plt.show()
fig, ax = plt.subplots()
fig.set_size_inches(xsize, ysize)

sns.countplot(x="phase", hue="target", data=train_meta_df, ax=ax)

plt.show()
fig, axes = plt.subplots(nrows=3, ncols=2)
fig.set_size_inches(2.0*xsize, 2.0*ysize)
axes = axes.flatten()

axes[0].plot(train_df["0"].values, marker="o", linestyle="none")
axes[0].set_title("Signal ID: 0")

axes[1].plot(train_df["2"].values, marker="o", linestyle="none")
axes[1].set_title("Signal ID: 1")

axes[2].plot(train_df["3"].values, marker="o", linestyle="none")
axes[2].set_title("Signal ID: 2")

axes[3].plot(train_df["4"].values, marker="o", linestyle="none")
axes[3].set_title("Signal ID: 3")

axes[4].plot(train_df["5"].values, marker="o", linestyle="none")
axes[4].set_title("Signal ID: 4")

axes[5].plot(train_df["6"].values, marker="o", linestyle="none")
axes[5].set_title("Signal ID: 5")

plt.show()
%%time

train_meta_df["signal_mean"] = train_df.agg(np.mean).values
train_meta_df["signal_sum"] = train_df.agg(np.sum).values
train_meta_df["signal_std"] = train_df.agg(np.std).values
train_meta_df.head()
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(2.0*xsize, 2.0*ysize)
axes = axes.flatten()

f, Pxx = welch(train_df["0"].values)
axes[0].plot(f, Pxx, marker="o", linestyle="none")
axes[0].set_title("Signal ID: 0")
axes[0].axhline(y=2.5, color="k", linestyle="--")

f, Pxx = welch(train_df["1"].values)
axes[1].plot(f, Pxx, marker="o", linestyle="none")
axes[1].set_title("Signal ID: 1")
axes[1].axhline(y=2.5, color="k", linestyle="--")

f, Pxx = welch(train_df["2"].values)
axes[2].plot(f, Pxx, marker="o", linestyle="none")
axes[2].set_title("Signal ID: 2")
axes[2].axhline(y=2.5, color="k", linestyle="--")

f, Pxx = welch(train_df["3"].values)
axes[3].plot(f, Pxx, marker="o", linestyle="none")
axes[3].set_title("Signal ID: 3")
axes[3].axhline(y=2.5, color="k", linestyle="--")

plt.show()
%%time

def welch_max_power_and_frequency(signal):
    f, Pxx = welch(signal)
    ix = np.argmax(Pxx)
    strong_count = np.sum(Pxx>2.5)
    avg_amp = np.mean(Pxx)
    sum_amp = np.sum(Pxx)
    std_amp = np.std(Pxx)
    median_amp = np.median(Pxx)
    return [Pxx[ix], f[ix], strong_count, avg_amp, sum_amp, std_amp, median_amp]

power_spectrum_summary = train_df.apply(welch_max_power_and_frequency, result_type="expand")
power_spectrum_summary = power_spectrum_summary.T.rename(columns={0:"max_amp", 1:"max_freq", 2:"strong_amp_count", 3:"avg_amp", 
                                                                  4:"sum_amp", 5:"std_amp", 6:"median_amp"})
power_spectrum_summary.head()
power_spectrum_summary.index = power_spectrum_summary.index.astype(int)
train_meta_df = train_meta_df.merge(power_spectrum_summary, left_on="signal_id", right_index=True)
train_meta_df.head()
X_cols = ["phase"] + train_meta_df.columns[4:].tolist()
X_cols
Fvals, pvals = f_classif(train_meta_df[X_cols], train_meta_df["target"])

print("F-value | P-value | Feature Name")
print("--------------------------------")

for i, col in enumerate(X_cols):
    print("%.4f"%Fvals[i]+" | "+"%.4f"%pvals[i]+" | "+col)
def mcc(y_true, y_pred, labels=None, sample_weight=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels, sample_weight=sample_weight).ravel()
    mcc = (tp*tn - fp*fn)/np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
    return mcc

mcc_scorer = make_scorer(mcc)

lgbm_classifier = lgbm.LGBMClassifier(boosting_type='gbdt', max_depth=-1, subsample_for_bin=200000, objective="binary", 
                                      class_weight=None, min_split_gain=0.0, min_child_weight=0.001, subsample=1.0, 
                                      subsample_freq=0, random_state=rand_seed, n_jobs=1, silent=True, importance_type='split')

param_distributions = {
    "num_leaves": randint(16, 48),
    "learning_rate": expon(),
    "reg_alpha": expon(),
    "reg_lambda": expon(),
    "colsample_bytree": uniform(0.25, 1.0),
    "min_child_samples": randint(10, 30),
    "n_estimators": randint(50, 250)
}

clf = RandomizedSearchCV(lgbm_classifier, param_distributions, n_iter=100, scoring=mcc_scorer, fit_params=None, n_jobs=1, iid=True, 
                         refit=True, cv=5, verbose=1, random_state=rand_seed, error_score=-1.0, return_train_score=True)
clf.fit(train_meta_df[X_cols], train_meta_df["target"])
print(clf.best_score_)
clf.best_estimator_
fig, ax = plt.subplots()
fig.set_size_inches(xsize, ysize)

lgbm.plot_importance(clf.best_estimator_, ax=ax)

plt.show()
