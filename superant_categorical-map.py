import numpy as np

import pandas as pd

import datetime as dt

import matplotlib.pyplot as plt

from sklearn.metrics import normalized_mutual_info_score, mutual_info_score

from tqdm import tqdm_notebook as tqdm

from itertools import combinations

import seaborn as sns

from functools import partial



import os
def astype_cat(dd, cols):

    for col in cols:

        if isinstance(col, tuple):

            col, idx1, idx2 = col

            for idx in range(idx1, idx2+1):

                full_col=col+str(idx)

                dd[full_col]=dd[full_col].astype("category")

        else:

            dd[col]=dd[col].astype("category")

            

dd=pd.read_csv("../input/train_transaction.csv")

astype_cat(dd, ["ProductCD", ("card", 1, 6), "addr1", "addr2", "P_emaildomain", "R_emaildomain", ("M", 1, 9)])



ddid=pd.read_csv("../input/train_identity.csv")

astype_cat(ddid, ["DeviceType", "DeviceInfo", ("id_", 12, 38)])



dd=dd.merge(ddid, "left", "TransactionID")



dd["datetime"]=(dd["TransactionDT"].apply(lambda x:dt.timedelta(seconds=x)+pd.Timestamp("2017-11-30")))



del ddid



dd.head()
cat_cols=dd.dtypes.loc[lambda x:x=="category"].index



def calc_scores(score_func):

    scores=[]

    for col1, col2 in tqdm(list(combinations(cat_cols, 2))):

        score=score_func(dd[col1].cat.codes, dd[col2].cat.codes)

        scores.append((col1, col2, score))

    scores=pd.DataFrame(scores, columns=["col1", "col2", "score"])

    

    scores_sym=pd.concat([scores, scores.rename(columns={"col1":"col2", "col2":"col1"})])

    

    return scores_sym
scores1=calc_scores(partial(normalized_mutual_info_score, average_method="arithmetic"))
sns.clustermap(scores1.pivot("col1", "col2", "score").fillna(scores1["score"].max()), figsize=(15,15));

display(scores1.sort_values("score", ascending=False).iloc[:20])
scores2=calc_scores(mutual_info_score)
sns.clustermap(scores2.pivot("col1", "col2", "score").fillna(scores2["score"].max())**(1/3), figsize=(15,15));

display(scores2.sort_values("score", ascending=False).iloc[:20])