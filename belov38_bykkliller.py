# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats  as stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
best = pd.read_csv("../input/round/submission.csv")

best.head()
sales = pd.read_csv("../input/m5-forecasting-uncertainty/sales_train_validation.csv")

sales.head()
sub = best.merge(sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]], on = "id")

sub["_all_"] = "Total"

sub.shape

qs = np.array([0.005,0.025,0.165,0.25, 0.5, 0.75, 0.835, 0.975, 0.995])

qs.shape
qs2 = np.log(qs/(1-qs))*.065



ratios = stats.norm.cdf(qs2)

ratios /= ratios[4]

ratios = pd.Series(ratios, index=qs)

ratios.round(3)
def quantile_coefs(q):

    return ratios.loc[q].values
def get_group_preds(pred, level):

    df = pred.groupby(level)[cols].sum()

    q = np.repeat(qs, len(df))

    df = pd.concat([df]*9, axis=0, sort=False)

    df.reset_index(inplace = True)

    df[cols] *= quantile_coefs(q)[:, None]

    if level != "id":

        df["id"] = [f"{lev}_X_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]

    else:

        df["id"] = [f"{lev.replace('_validation', '')}_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]

    df = df[["id"]+list(cols)]

    return df
def get_couple_group_preds(pred, level1, level2):

    df = pred.groupby([level1, level2])[cols].sum()

    q = np.repeat(qs, len(df))

    df = pd.concat([df]*9, axis=0, sort=False)

    df.reset_index(inplace = True)

    df[cols] *= quantile_coefs(q)[:, None]

    df["id"] = [f"{lev1}_{lev2}_{q:.3f}_validation" for lev1,lev2, q in 

                zip(df[level1].values,df[level2].values, q)]

    df = df[["id"]+list(cols)]

    return df
levels = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "_all_"]

couples = [("state_id", "item_id"),  ("state_id", "dept_id"),("store_id","dept_id"),

                            ("state_id", "cat_id"),("store_id","cat_id")]

cols = [f"F{i}" for i in range(1, 29)]
df = []

for level in levels :

    df.append(get_group_preds(sub, level))

for level1,level2 in couples:

    df.append(get_couple_group_preds(sub, level1, level2))

df = pd.concat(df, axis=0, sort=False)

df.reset_index(drop=True, inplace=True)

df = pd.concat([df,df] , axis=0, sort=False)

df.reset_index(drop=True, inplace=True)

df.loc[df.index >= len(df.index)//2, "id"] = df.loc[df.index >= len(df.index)//2, "id"].str.replace(

                                    "_validation$", "_evaluation")



df.shape
df.to_csv("submission.csv", index = False)