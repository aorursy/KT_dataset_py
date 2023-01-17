from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np 

import os





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
submission = pd.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv")

train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")

test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")



labels = train.pop('target')

train_id = train.pop("id")

test_id = test.pop("id")
print(train.shape)

print(test.shape)
sns.countplot(labels)

plt.title("labels counts")

plt.show()
def count_plot(data, columns):

    

    fig, axes = plt.subplots(nrows=len(columns), figsize=(10,10))

    fig.subplots_adjust(hspace=0.3)



    for (i, col) in enumerate(columns):

        sns.countplot(data[col], ax=axes[i])

        axes[i].set_xlabel(None)

        axes[i].set_ylabel("count " + str(col), fontsize=12)

    

    plt.show()
bin_col = ['bin_0', 'bin_2', 'bin_3', 'bin_4']



count_plot(train, bin_col)
nom_col = ['nom_0', 'nom_2', 'nom_3', 'nom_4']



count_plot(train, nom_col)
ord_col = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4']



count_plot(train, ord_col)
labels = labels.values
data = pd.concat([train, test])
data["ord_5a"] = data["ord_5"].str[0]

data["ord_5b"] = data["ord_5"].str[1]
data.drop(["bin_0", "ord_5"], axis=1, inplace=True)
columns = [i for i in data.columns]



dummies = pd.get_dummies(data,

                         columns=columns,

                         drop_first=True,

                         sparse=True)



del data
train = dummies.iloc[:train.shape[0], :]

test = dummies.iloc[train.shape[0]:, :]



del dummies
print(train.shape)

print(test.shape)
train = train.sparse.to_coo().tocsr()

test = test.sparse.to_coo().tocsr()



train = train.astype("float32")

test = test.astype("float32")
lr = LogisticRegression(C=0.1338,

                        solver="lbfgs",

                        tol=0.0003,

                        max_iter=5000)



lr.fit(train, labels)



lr_pred = lr.predict_proba(train)[:, 1]

score = roc_auc_score(labels, lr_pred)



print("score: ", score)
submission["id"] = test_id

submission["target"] = lr.predict_proba(test)[:, 1]
submission.head()
submission.to_csv("submission.csv", index=False)