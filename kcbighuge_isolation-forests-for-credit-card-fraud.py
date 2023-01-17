# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
PATH = "../input/"
import h2o
h2o.init()

df = h2o.import_file(PATH+"creditcard.csv")
print(df.shape)
df.head()
seed = 311
ntrees = 80
isoforest = h2o.estimators.H2OIsolationForestEstimator(
    ntrees=ntrees, seed=seed)
isoforest.train(x=df.col_names[0:31], training_frame=df)
predictions = isoforest.predict(df)

predictions
predictions.shape
predictions.cor()
quantile = 0.95
quantile_frame = predictions.quantile([quantile])
quantile_frame
threshold = quantile_frame[0, "predictQuantiles"]
predictions["predicted_class"] = predictions["predict"] > threshold
predictions["class"] = df["Class"]
predictions
#%matplotlib notebook
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np


def get_auc(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score


def get_aucpr(labels, scores):
    precision, recall, th = precision_recall_curve(labels, scores)
    aucpr_score = np.trapz(recall, precision)
    return precision, recall, aucpr_score


def plot_metric(ax, x, y, x_label, y_label, plot_label, style="-"):
    ax.plot(x, y, style, label=plot_label)
    ax.legend()
    
    ax.set_ylabel(x_label)
    ax.set_xlabel(y_label)


def prediction_summary(labels, predicted_score, predicted_class, info, plot_baseline=True, axes=None):
    if axes is None:
        axes = [plt.subplot(1, 2, 1), plt.subplot(1, 2, 2)]

    fpr, tpr, auc_score = get_auc(labels, predicted_score)
    plot_metric(axes[0], fpr, tpr, "False positive rate",
                "True positive rate", "{} AUC = {:.4f}".format(info, auc_score))
    if plot_baseline:
        plot_metric(axes[0], [0, 1], [0, 1], "False positive rate",
                "True positive rate", "baseline AUC = 0.5", "r--")

    precision, recall, aucpr_score = get_aucpr(labels, predicted_score)
    plot_metric(axes[1], recall, precision, "Recall",
                "Precision", "{} AUCPR = {:.4f}".format(info, aucpr_score))
    if plot_baseline:
        thr = sum(labels)/len(labels)
        plot_metric(axes[1], [0, 1], [thr, thr], "Recall",
                "Precision", "baseline AUCPR = {:.4f}".format(thr), "r--")

    plt.show()
    return axes


def figure():
    fig_size = 4.5
    f = plt.figure()
    f.set_figheight(fig_size)
    f.set_figwidth(fig_size*2)
h2o_predictions = predictions.as_data_frame()

plt.figure(figsize=(10,4))
axes = prediction_summary(
    h2o_predictions["class"], h2o_predictions["predict"], h2o_predictions["predicted_class"], 
    "h2o")
from sklearn.ensemble import IsolationForest

df_pandas = df.as_data_frame()
df_train_pandas = df_pandas.iloc[:, :30]

x = IsolationForest(random_state=seed, contamination=(1-quantile),
                    n_estimators=ntrees, behaviour="new").fit(df_train_pandas)

iso_predictions = x.predict(df_train_pandas)
iso_score = x.score_samples(df_train_pandas)

sk_predictions = pd.DataFrame({
    "predicted_class": list(map(lambda x: 1*(x == -1), iso_predictions)),
    "class": h2o_predictions["class"],
    "predict": -iso_score
})

sk_predictions.head()
# Evaluate the sklearn model
figure()
axes = prediction_summary(
    sk_predictions["class"], sk_predictions["predict"], sk_predictions["predicted_class"], 
    "sklearn")
from tqdm import tqdm_notebook

def stability_check(train_predict_fn, x, y, ntimes=8):
    scores = ["AUC", "AUCPR"]
    scores = {key: [] for key in scores}
    seeds = np.linspace(1, (2**32) - 1, ntimes).astype(int)
    for seed in tqdm_notebook(seeds):
        predictions = train_predict_fn(x, int(seed))
        _, _, auc_score = get_auc(y, predictions)
        _, _, aucpr_score = get_aucpr(y, predictions)

        scores["AUC"].append(auc_score)
        scores["AUCPR"].append(aucpr_score)
        print("Finished training with random seed: {}".format(seed))

    return pd.DataFrame(scores)


def iso_forests_h2o(data, seed):
    isoforest = h2o.estimators.H2OIsolationForestEstimator(
        ntrees=ntrees, seed=seed)
    isoforest.train(x=data.col_names, training_frame=data)
    preds = isoforest.predict(data)
    return preds.as_data_frame()["predict"]


def iso_forests_sklearn(data, seed):
    iso = IsolationForest(random_state=seed, n_estimators=ntrees,
                          behaviour="new", contamination=(1-quantile))
    iso.fit(data)
    iso_score = iso.score_samples(data)
    return -iso_score
h2o_check = stability_check(iso_forests_h2o, df[:30], h2o_predictions["class"])
sklearn_check = stability_check(
    iso_forests_sklearn, df_train_pandas, sk_predictions["class"])

sklearn_check.join(h2o_check, rsuffix="_h2o", lsuffix="_sklearn").describe()