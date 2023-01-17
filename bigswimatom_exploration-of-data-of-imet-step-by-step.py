import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))
# View

import matplotlib.pyplot as plt

from PIL import Image, ImageFilter

%matplotlib inline
from collections import Counter
# Load tabular data

train_df = pd.read_csv("../input/train.csv", index_col=0)

labels_df = pd.read_csv("../input/labels.csv", index_col=0)

sample_df = pd.read_csv("../input/sample_submission.csv", index_col=0)
train_df.describe()
labels_df.describe()
sample_df.describe()
flatten = lambda x: [z for y in x for z in (flatten(y) if hasattr(y, '__iter__') and not isinstance(y, str) else (y, ))]
attribute_dist = pd.Series(flatten(list(train_df["attribute_ids"].map(lambda x: x.split(" "))))).value_counts()

attribute_dist = pd.DataFrame(attribute_dist, columns=["Count"])

attribute_dist = attribute_dist.reset_index()

attribute_dist.columns = ["attribute_id", "Count"]

attribute_dist["attribute_name"] = attribute_dist["attribute_id"].map(lambda x: labels_df.loc[int(x)].values[0])

attribute_dist["ratio"] = attribute_dist["Count"] / attribute_dist["Count"].sum()

attribute_dist["cumsum"] = attribute_dist["ratio"].cumsum()

attribute_dist.columns = ["attribute_id", "Count", "attribute_name", "ratio", "cumsum"]

attribute_dist = attribute_dist[["attribute_id", "attribute_name", "Count", "ratio", "cumsum"]]


attribute_dist
plt.figure(figsize=(16, 7))

plt.plot(attribute_dist["cumsum"])

plt.xlabel("attribute_dist")

plt.ylabel("cumsum")
# Rank A

rank_A = attribute_dist[attribute_dist["cumsum"] <= 0.7]

# Rank B

rank_B = attribute_dist[(attribute_dist["cumsum"] > 0.7) & (attribute_dist["cumsum"] <= 0.9)]

# Rank C

rank_C = attribute_dist[attribute_dist["cumsum"] > 0.9]
len(rank_A), len(rank_B), len(rank_C)
attribute_dist.to_csv("attribute_distribution.csv")

attribute_dist