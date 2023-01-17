from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import os
data_dir = os.path.join("..", "input")
train_transaction = pd.read_csv(os.path.join(data_dir, "train_transaction.csv"), index_col=0)

train_identity = pd.read_csv(os.path.join(data_dir, "train_identity.csv"), index_col=0)
test_transaction = pd.read_csv(os.path.join(data_dir, "test_transaction.csv"), index_col=0)

test_identity = pd.read_csv(os.path.join(data_dir, "test_identity.csv"), index_col=0)
train = train_transaction.join(train_identity)

test = test_transaction.join(test_identity)
sample = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
cols = ["ProductCD", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"]

cols.extend([col for col in train.columns if col.startswith(("card", "addr", "M", "id"))])

train[cols] = train[cols].astype("str")

test[cols] = test[cols].astype("str")
def plot_fts(fts, n_cols=3, m_cat=10):

    n_fts = len(fts)

    n_rows = -(-n_fts // n_cols) # ceiling division

    to_del = (n_cols - (n_fts % n_cols)) % n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))

    plt.subplots_adjust(wspace=0.3, hspace=0.45)

    axes = (axes if type(axes) is np.ndarray else np.array(axes)).ravel()

    strp = lambda l: f"{float(l):g}" if l.replace(".", "").isdigit() else l[:7]

    np_col = train.select_dtypes(np.number).columns

    for i, ft in enumerate(fts):

        ax1 = axes[i]

        ax2 = ax1.twinx()

        if ft in np_col:

            cut, bins = pd.cut(train[ft], 10, retbins=True, right=False)

            ax1.hist(train[ft], bins, color="#A8DBA8", edgecolor="k", log=True, zorder=2)

            ax1.tick_params("x", pad=2)

            ax1.grid(zorder=0)

            ctrs = (bins[:-1] + bins[1:]) / 2 

            vals = train.groupby(cut)["isFraud"].mean()

            ax2.plot(ctrs, vals, marker="o", c="#0B486B", lw=2)

        else:

            cts = train[ft].value_counts().nlargest(m_cat)

            ax1.bar(cts.index, cts.values, width=1, color="#a6cee3", edgecolor="k", log=True, zorder=2)

            ax1.tick_params("x", labelrotation=45, pad=0)

            ax1.grid(zorder=0)

            vals = train.groupby(ft)["isFraud"].mean().loc[cts.index]

            ax2.plot(vals, marker="o", c="#0B486B", lw=2)

            ax2.set_xticklabels(list(map(strp, cts.index)))

        ax1.minorticks_off()

        ax2.minorticks_off()

        ax1.set_ylim(1, 10**6)

        ax1.set_title(ft, loc="right")

        ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    for i in range(to_del, 0, -1):

        fig.delaxes(axes[-i])

    plt.show()
plot_fts(train.drop("isFraud", axis=1).columns)