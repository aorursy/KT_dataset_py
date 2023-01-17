import numpy as np

import pandas as pd

from tqdm import tqdm

import matplotlib.pyplot as plt

pd.set_option("max_columns",300)

pd.set_option("max_rows",1020)
train_df = pd.read_csv("../input/train.csv")

labels_df = pd.read_csv("../input/labels.csv")
train_df["attribute_ids"]=train_df["attribute_ids"].apply(lambda x:[int(x) for x in x.split(" ")])

train_df["id"]=train_df["id"].apply(lambda x:x+".png")

train_df.head()
for ix in tqdm(range(1103)):

    train_df["label_"+str(ix)] = train_df["attribute_ids"].map(lambda x: 1 if ix in x else 0)
train_df.head()
labels = ["label_{}".format(i) for i in range(1103)]

summary = train_df[labels].sum()
labels_df["count"] = summary.values
labels_df.sort_values("count")
fig = plt.figure(figsize=(20,5))

ax = fig.add_subplot(1, 1, 1)

ax.set_xticks([x for x in range(0,1000,20)]) 

plt.xlabel('train num per label')

plt.hist(summary, range=(0, 1000), bins=50);
for num in [5, 10, 50, 100]: 

    print("under {0}: {1}".format(num, (summary<=num).sum()))