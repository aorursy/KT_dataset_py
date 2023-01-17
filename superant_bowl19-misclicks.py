import numpy as np 

import pandas as pd

import json

from pandas.io.json import json_normalize

import seaborn as sns

import matplotlib.pyplot as plt

from operator import attrgetter
dtypes = {

        "title": "category",

        "event_id": "category",

        "event_count": "int16",

        "event_code": "int16",

        "game_time": "int32",

        "title": "category",

        "type": "category",

        "world": "category",

        "installation_id": "category",

    }



D = pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv", dtype=dtypes, parse_dates=["timestamp"])

D.head()
L=pd.read_csv("/kaggle/input/data-science-bowl-2019/train_labels.csv")

D["acc"]=D["game_session"].map(L.set_index("game_session")["accuracy_group"])
def add_misclick_acc(dd, bins=20):

    # Add coordinatess

    dd=pd.concat([dd, dd.query("event_code==4070 and type=='Assessment'")["event_data"].str.extract('"coordinates":{"x":(?P<x>[0-9]+),"y":(?P<y>[0-9]+),').astype(int)], axis=1)



    # Bin coordinates

    for col in ["x", "y"]:

        dd[f"{col}_bin"] = pd.cut(dd[col], bins=bins).apply(attrgetter("left"))



    # Calc and add rate

    misclick_acc = dd.query("event_code==4070 and type=='Assessment'").groupby(["x_bin", "y_bin", "title"], observed=True)["acc"].mean().rename("misclick_acc")

    dd = dd.merge(misclick_acc.reset_index(), on=["x_bin", "y_bin", "title"], how="left")

    

    return dd



dd = add_misclick_acc(D)

dd.dropna(subset=["misclick_acc"])
titles=dd.query("type=='Assessment'")["title"].unique()

for title in titles:

    dd_plot = dd.query("title==@title")

    plt.hexbin("x", "y", dd_plot["acc"], data=dd_plot, cmap="inferno", gridsize=20);

    plt.colorbar()

    plt.title(title)

    plt.show()