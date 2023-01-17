import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
input_data_df = {}



for dirname, _, filenames in os.walk("/kaggle/input"):

    for filename in filenames:

        base = os.path.splitext(filename)[0]

        input_data_df[base] = pd.read_csv(os.path.join(dirname, filename))
calendar_df = input_data_df["calendar"].copy()

calendar_df.head(5)
def get_snap_means(filter):

    snap_days = calendar_df[filter]["d"].values

    data_df = input_data_df["sales_train_validation"].copy()

    feats = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] + list(snap_days)

    data_df = data_df.loc[:, feats]

    data_df.dropna(how="all", axis=1, inplace=True)



    snap_days_in_data = []

    for c in data_df.columns:

        if c.startswith("d_"):

            snap_days_in_data.append(c)



    agg_dicts = {}

    for d in snap_days_in_data:

        agg_dicts[d] = np.sum



    snap_means_df = data_df.groupby(["cat_id", "state_id"]).agg(agg_dicts)

    snap_means_df = pd.DataFrame(snap_means_df.T.mean())

    

    return snap_means_df



snap_means_df = get_snap_means(calendar_df["snap_TX"] == 0).drop([0], axis=1)

for category in ["CA", "TX", "WI"]:

    for w in [0, 1]:

        snap_means_df = pd.merge(snap_means_df, get_snap_means(calendar_df["snap_{}".format(category)] == w), left_index=True, right_index=True, how="left")

        snap_means_df.rename(columns={0: "{}/ snap_{}".format("w" if w else "wo", category)}, inplace=True)
snap_means_df
fig = plt.figure(figsize=(12.0, 12.0))



for i, category in enumerate(["FOODS", "HOBBIES", "HOUSEHOLD"]):

    for j, state in enumerate(["CA", "TX", "WI"]):

        ax = fig.add_subplot(33*10 + i * 3 + j + 1)

        index = np.arange(3)

        bar_width = 0.35

        labels = snap_means_df.loc[category].index

        ax.bar(index, snap_means_df.loc[category]["wo/ snap_{}".format(state)], bar_width,

               color="b", label="wo/ snap_{}".format(state))

        ax.bar(index + bar_width, snap_means_df.loc[category]["w/ snap_{}".format(state)], bar_width,

               color="r", label="w/ snap_{}".format(state))

        ax.set_title("Number of {} (snap_{})".format(category, state))

        ax.set_xlabel("state_id")

        ax.set_ylabel("Average Number of Items")

        ax.set_xticks(index + bar_width / 2)

        ax.set_xticklabels(labels)

        ax.legend()



fig.tight_layout()