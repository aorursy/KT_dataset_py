import pandas as pd 

import datetime as dt

import matplotlib as mpl

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np
def astype_cat(dd, cols):

    for col in cols:

        if isinstance(col, tuple):

            col, idx1, idx2 = col

            for idx in range(idx1, idx2+1):

                full_col=col+str(idx)

                dd[full_col]=dd[full_col].astype("category")

        else:

            dd[col]=dd[col].astype("category")

            

def load(trans_filename, id_filename):

    dd=pd.read_csv(trans_filename)

    astype_cat(dd, ["ProductCD", ("card", 1, 6), "addr1", "addr2", "P_emaildomain", "R_emaildomain", ("M", 1, 9)])



    ddid=pd.read_csv(id_filename)

    astype_cat(ddid, ["DeviceType", "DeviceInfo", ("id_", 12, 38)])



    dd=dd.merge(ddid, "left", "TransactionID")



    #dd["datetime"]=(dd["TransactionDT"].apply(lambda x:dt.timedelta(seconds=x)+pd.Timestamp("2017-11-30")))



    return dd



dd=load("../input/train_transaction.csv", "../input/train_identity.csv")

ddtest=load("../input/test_transaction.csv", "../input/test_identity.csv")



dd.head()
cat_cols=dd.dtypes.loc[lambda x:x=="category"].index



sns.set_palette("pastel")



def plot_cat(dd, col):

    mean_fraud=dd["isFraud"].mean()



    plt.figure(figsize=(15,10))



    max_show_cats=30

    cnts=dd[col].value_counts(normalize=True, dropna=False)

    plot_cnts=cnts.iloc[:max_show_cats]

    plt.bar(range(len(plot_cnts)), plot_cnts, width=0.9, tick_label=plot_cnts.index)

    plt.xticks(rotation=45, ha="right")

    plt.ylabel("Category rate")

    plt.grid(False)

    

    test_cnts=ddtest[col].value_counts(normalize=True)

    plt.step(range(len(plot_cnts)), test_cnts.reindex(plot_cnts.index).fillna(0), c="b", where="mid", label="Test data")

    plt.legend()

    

    fraud_rate=dd.groupby(col)["isFraud"].mean()

    if dd[col].isnull().any():

        fraud_rate[np.nan]=dd.loc[dd[col].isnull(), "isFraud"].mean()

        

    ax_fraud=plt.gca().twinx()

    ax_fraud.stem(range(len(plot_cnts)), fraud_rate.reindex(plot_cnts.index), linefmt="k:", markerfmt="ko")

    ax_fraud.set_ylabel("Fraud rate", color="k")

    ax_fraud.grid(False)

    ax_fraud.axhline(mean_fraud, ls="--", c="k")

    ax_fraud.set_ylim(bottom=0)



    title=[col]

    if len(cnts)>len(plot_cnts):

        title.append(f"({plot_cnts.sum():.0%} of data; {len(plot_cnts)}/{len(cnts)} cats)")

    plt.title(" ".join(title))

    plt.show()

    

    

for cat in cat_cols:

    plot_cat(dd, cat)
mean_fraud=dd["isFraud"].mean()



def plot_val(dd, col, logbin=True):

    edges=np.histogram_bin_edges(list(dd[col].dropna())+list(ddtest[col].dropna()), bins="doane")

    plt.figure(figsize=(15,10))

    plt.hist(dd[col], bins=edges, log=logbin)

    plt.hist(ddtest[col], bins=edges, log=logbin, histtype="step", color="b")



    fraud_rate=dd.groupby(pd.cut(dd[col], bins=edges))["isFraud"].mean()

    ax_fraud=plt.gca().twinx()

    ax_fraud.stem(fraud_rate.index.map(lambda x:x.mid), fraud_rate, linefmt="k:", markerfmt="ko")

    ax_fraud.set_ylabel("Fraud rate", color="k")

    ax_fraud.grid(False)

    ax_fraud.axhline(mean_fraud, ls="--", c="k")



    plt.title(f"{col}")

    plt.show()



for col in sorted(dd.columns.difference(set(cat_cols)|{"datetime", "isFraud"})):

    plot_val(dd, col)