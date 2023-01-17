import os



print(os.listdir("../input"))
import numpy as np

import pandas as pd



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

sample_submission = pd.read_csv("../input/sample_submission.csv")



train.head(3)
train["target"].value_counts()
import matplotlib.pyplot as plt



fig = plt.figure(figsize=(20,3))



for i in range(5):

    ax = fig.add_subplot(1,5,i+1)

    train["{}".format(i)].hist(bins=25,ax=ax)

    ax.set_title("feature {}".format(i))



plt.tight_layout()

plt.show()
train.isnull().any().any()
corrs = train.corr().abs().unstack().sort_values().reset_index()

corrs = corrs[corrs["level_0"] != corrs["level_1"]]

corrs = corrs.loc[corrs.index % 2 == 1].reset_index(drop=True)
corrs.head(5)
corrs.tail(5)
fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(111)



ax.matshow(train.drop("id",axis=1).corr())

ax.set_xticklabels(train.columns)

ax.set_yticklabels(train.columns)



plt.show()