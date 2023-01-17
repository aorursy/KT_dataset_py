import collections

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
TRAIN_LOC = "/kaggle/input/student-shopee-code-league-sentiment-analysis/train.csv"

TEST_LOC = "/kaggle/input/student-shopee-code-league-sentiment-analysis/test.csv"
df_train = pd.read_csv(TRAIN_LOC)

df_test = pd.read_csv(TEST_LOC)
df_train
labels, values = zip(*sorted(collections.Counter(df_train["rating"]).items()))

values = [v/len(df_train) for v in values]

indexes = np.arange(len(labels))



plt.bar(indexes, values, 0.5)

plt.xticks(indexes, labels)

plt.show()
# give some default rating

df_test["rating"] = 5
df_test[["review_id", "rating"]].to_csv("submission.csv", index=False)
!head -3 /kaggle/input/student-shopee-code-league-sentiment-analysis/sampleSubmission.csv
!head -3 submission.csv