# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/instant-gratification/train.csv")

test_df = pd.read_csv("/kaggle/input/instant-gratification/test.csv")
train_df.head()
test_df.head()
features = [c for c in train_df.columns.values if c not in ["id", "target"]]
train_df[features].describe()
test_df.describe()
plt.figure(figsize = (16,6))

sns.distplot(train_df[features].mean(axis = "columns"), color = "magenta", label = "train", kde = True, bins = 120)

sns.distplot(test_df[features].mean(axis = "columns"), color = "darkblue", label = "test", kde = True, bins = 120)

plt.legend()

plt.show()
plt.figure(figsize = (16,6))

sns.distplot(train_df[features].mean(axis = "rows"), color = "magenta", label = "train", kde = True, bins = 120)

sns.distplot(test_df[features].mean(axis = "rows"), color = "darkblue", label = "test", kde = True, bins = 120)

plt.legend()

plt.show()
plt.figure(figsize = (16,6))

sns.distplot(train_df[features].std(axis = "columns"), color = "magenta", label = "train", kde = True, bins = 120)

sns.distplot(test_df[features].std(axis = "columns"), color = "darkblue", label = "test", kde = True, bins = 120)

plt.legend()

plt.show()
plt.figure(figsize = (16,6))

sns.distplot(train_df[features].std(axis = "rows"), color = "magenta", label = "train", kde = True, bins = 120)

sns.distplot(test_df[features].std(axis = "rows"), color = "darkblue", label = "test", kde = True, bins = 120)

plt.legend()

plt.show()
plt.figure(figsize = (16,6))

sns.distplot(train_df[features].max(axis = "columns"), color = "magenta", label = "train", kde = True, bins = 120)

sns.distplot(test_df[features].max(axis = "columns"), color = "darkblue", label = "test", kde = True, bins = 120)

plt.legend()

plt.show()
plt.figure(figsize = (16,6))

sns.distplot(train_df[features].max(axis = "rows"), color = "magenta", label = "train", kde = True, bins = 120)

sns.distplot(test_df[features].max(axis = "rows"), color = "darkblue", label = "test", kde = True, bins = 120)

plt.legend()

plt.show()
plt.figure(figsize = (16,6))

sns.distplot(train_df[features].min(axis = "columns"), color = "magenta", label = "train", kde = True, bins = 120)

sns.distplot(test_df[features].min(axis = "columns"), color = "darkblue", label = "test", kde = True, bins = 120)

plt.legend()

plt.show()
plt.figure(figsize = (16,6))

sns.distplot(train_df[features].min(axis = "rows"), color = "magenta", label = "train", kde = True, bins = 120)

sns.distplot(test_df[features].min(axis = "rows"), color = "darkblue", label = "test", kde = True, bins = 120)

plt.legend()

plt.show()
for df in [train_df, test_df]:

    df["sum"] = df[features].sum(axis = 1)

    df["mean"] = df[features].mean(axis = 1)

    df["std"] = df[features].std(axis = 1)

    df["max"] = df[features].max(axis = 1)

    df["min"] = df[features].min(axis = 1)

    df["median"] = df[features].median(axis = 1)
train_df[train_df.columns[258:]].head()
test_df[test_df.columns[257:]].head()
t0 = train_df[train_df["target"] == 0]

t1 = train_df[train_df["target"] == 1]
n_rows = 2

n_cols = 3

feats = test_df[test_df.columns[257:]].columns



fig, axs = plt.subplots(n_rows, n_cols, figsize = (n_cols * 4, n_rows * 3))



for r in range(n_rows):

    for c in range(n_cols):

        i = r * n_cols + c

        

        if  i < len(feats):

            sns.kdeplot(t0[feats[i]], ax = axs[r][c])

            sns.kdeplot(t1[feats[i]], ax = axs[r][c])



plt.tight_layout()

plt.show()
n_rows = 2

n_cols = 3

feats = test_df[test_df.columns[257:]].columns



fig, axs = plt.subplots(n_rows, n_cols, figsize = (n_cols * 4, n_rows * 3))



for r in range(n_rows):

    for c in range(n_cols):

        i = r * n_cols + c

        

        if  i < len(feats):

            sns.kdeplot(train_df[feats[i]], ax = axs[r][c])

            sns.kdeplot(test_df[feats[i]], ax = axs[r][c])



plt.tight_layout()

plt.show()
for feature in features:

    train_df["r_" + feature] = np.round(train_df[feature],1)

    test_df["r_" + feature] = np.round(test_df[feature],1)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(train_df, train_df["target"], test_size = 0.25)



print(x_train.shape, y_train.shape)

print(x_test.shape, y_test.shape)
features = [c for c in train_df.columns.values if c not in ["id", "target"]]

len(features)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import r2_score



rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(x_train[test_df.columns[1:]], y_train)

predicts = rfc.predict(x_test[test_df.columns[1:]])



print("accuracy : ", accuracy_score(predicts, y_test))
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import r2_score



gbc = GradientBoostingClassifier()

gbc.fit(x_train[test_df.columns[1:]], y_train)

predicts = gbc.predict(x_test[test_df.columns[1:]])



print("accuracy : ", accuracy_score(predicts, y_test))
predictions = rfc.predict(test_df[test_df.columns[1:]])



pd.DataFrame({"id" : test_df["id"], "prediction" : predictions})