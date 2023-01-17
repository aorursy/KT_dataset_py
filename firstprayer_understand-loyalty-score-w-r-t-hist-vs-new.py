# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from matplotlib import pyplot as plt

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
hist_tran = pd.read_csv("../input/historical_transactions.csv")
new_tran = pd.read_csv("../input/new_merchant_transactions.csv")

print("loading done")
overlap_card_train_hist = set(train["card_id"]) & set(hist_tran["card_id"])
overlap_card_train_new = set(train["card_id"]) & set(new_tran["card_id"])

overlap_card_test_hist = set(test["card_id"]) & set(hist_tran["card_id"])
overlap_card_test_new = set(test["card_id"]) & set(new_tran["card_id"])

print(len(overlap_card_train_hist), len(overlap_card_train_new), len(train["card_id"]))
print(len(overlap_card_test_hist), len(overlap_card_test_new), len(test["card_id"]))
train["card_in_new"] = pd.Series([i in overlap_card_train_new for i in train["card_id"]], index=train.index)

scores_not_in_new = train[train["card_in_new"] == False]["target"]
scores_not_in_new.describe()
scores_in_new = train[train["card_in_new"] == True]["target"]
scores_in_new.describe()

from matplotlib import pyplot as plt
plt.figure(figsize=(12, 5))
plt.hist(scores_not_in_new.values, bins=200)
plt.title('Histogram target counts for scores_not_in_new')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()
plt.figure(figsize=(12, 5))
plt.hist(scores_in_new.values, bins=200)
plt.title('Histogram target counts for scores_in_new')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()
train_cleaned = train[train["target"] > -30]

scores_not_in_new_cleaned = train_cleaned[train_cleaned["card_in_new"] == False]["target"]
print(scores_not_in_new_cleaned.describe())

scores_in_new_cleaned = train_cleaned[train_cleaned["card_in_new"] == True]["target"]
print(scores_in_new_cleaned.describe())

from matplotlib import pyplot as plt
plt.figure(figsize=(12, 5))
plt.hist(scores_not_in_new_cleaned.values, bins=200)
plt.title('Histogram target counts for scores_not_in_new_cleaned')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()

from matplotlib import pyplot as plt
plt.figure(figsize=(12, 5))
plt.hist(scores_in_new_cleaned.values, bins=200)
plt.title('Histogram target counts for scores_in_new_cleaned')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()

print(
    len(scores_in_new_cleaned[scores_in_new_cleaned == 0.0]),
    len(scores_in_new_cleaned),
    len(scores_in_new_cleaned[scores_in_new_cleaned == 0.0]) / len(scores_in_new_cleaned),
)
print(
    len(scores_not_in_new_cleaned[scores_not_in_new_cleaned == 0.0]),
    len(scores_not_in_new_cleaned),
    len(scores_not_in_new_cleaned[scores_not_in_new_cleaned == 0.0]) / len(scores_not_in_new_cleaned)
)

card_id_to_new_merchants = {}
for _, row in new_tran.iterrows():
    card_id_to_new_merchants.setdefault(row.card_id, set())
    card_id_to_new_merchants[row.card_id].add(row.merchant_id)
    
train_cleaned["new_merchant_count"] = pd.Series(
    [len(card_id_to_new_merchants.get(c, set())) for c in train_cleaned["card_id"]],
    index=train_cleaned.index)


train_cleaned_in_new = train_cleaned[train_cleaned["card_in_new"] == True]
train_cleaned_in_new_and_zero_score = train_cleaned_in_new[train_cleaned_in_new["target"] == 0]
train_cleaned_in_new_and_zero_score["new_merchant_count"].describe()
plt.figure(figsize=(12, 5))
plt.hist(train_cleaned_in_new_and_zero_score["new_merchant_count"].values, bins=200)
plt.title('Histogram new merchant counts for train_cleaned_in_new_and_zero_score')
plt.xlabel('Count')
plt.ylabel('New merchant count')
plt.show()
train_cleaned_in_new_and_nonzero_score = train_cleaned_in_new[train_cleaned_in_new["target"] != 0]
print(train_cleaned_in_new_and_nonzero_score["new_merchant_count"].describe())

plt.figure(figsize=(12, 5))
plt.hist(train_cleaned_in_new_and_nonzero_score["new_merchant_count"].values, bins=200)
plt.title('Histogram new merchant counts for train_cleaned_in_new_and_nonzero_score')
plt.xlabel('Count')
plt.ylabel('New merchant count')
plt.show()
plt.scatter(
    train_cleaned_in_new["new_merchant_count"].values,
    train_cleaned_in_new["target"].values
)
max_merchant_bucket = train_cleaned_in_new["new_merchant_count"].max()
expectations = [
    train_cleaned_in_new[train_cleaned_in_new["new_merchant_count"] == i]["target"].mean()
     for i in range(max_merchant_bucket + 1)
]
expectations = [
    -5 if np.isnan(v) else v
    for v in expectations
]
# plt.plot(
#     range(max_merchant_bucket + 1),
#     [train_cleaned_in_new[train_cleaned_in_new["new_merchant_count"] == i].mean()
#      for i in range(max_merchant_bucket + 1)]
# )
plt.scatter(
    range(max_merchant_bucket + 1),
    expectations
)