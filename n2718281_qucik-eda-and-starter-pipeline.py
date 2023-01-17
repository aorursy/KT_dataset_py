import numpy as np

import pandas as pd

import category_encoders as ce

import matplotlib.pyplot as plt



from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

from sklearn.pipeline import make_pipeline

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

from sklearn.dummy import DummyClassifier
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')
train.head()
train.isna().sum().sum()
train.nunique()
for c in train.columns[1:]:

    print("="*5)

    print(c)

    print(train[c].value_counts().sort_values())
lc_feats = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'day', 'month', 'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']

hc_feats = ['ord_5', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
for col in lc_feats:

    

    x = list(train.groupby(col)['target'].mean().sort_index().index)

    y1 = train.groupby(col)['target'].mean().sort_index().values

    y2 = train[col].value_counts().sort_index().values

    

    fig, ax1 = plt.subplots()

    color = 'tab:blue'

    ax1.set_xlabel(col)

    ax1.set_ylabel('percentage of positive in the category', color=color)

    ax1.plot(x, y1, marker="o", color=color)

    ax1.tick_params(axis='y', labelcolor=color)



    ax2 = ax1.twinx()

    color = 'tab:red'

    ax2.bar(x,y2, color=color, alpha=0.2)

    ax2.set_ylabel('count', color=color)

    ax2.tick_params(axis='y', labelcolor=color)



    fig.tight_layout()

    plt.show()
pipeline = make_pipeline(

    ce.TargetEncoder(cols=hc_feats),               

    ce.OneHotEncoder(cols=lc_feats),

    LogisticRegression(solver="saga", penalty="l2", max_iter = 5000)

)
kfold = StratifiedKFold(n_splits=5, shuffle= True, random_state=42)
X_train = train.drop(["target", "id"], axis=1)

y_train = train["target"]
%%time

scores = cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring='roc_auc', n_jobs=-1, verbose=1)

print(scores)
scores.mean()