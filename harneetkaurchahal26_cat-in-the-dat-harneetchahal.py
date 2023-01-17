# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

from sklearn.linear_model import LogisticRegression

import scipy

from tqdm import tqdm_notebook as tqdm
dd0=pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")

ddtest0=pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")

ddall=dd0.append(ddtest0, sort=False)

num_train=len(dd0)

ddall.head()
drop_cols=["bin_0"]



# Split 2 Letters; This is the only part which is not generic and would actually require data inspection

ddall["ord_5a"]=ddall["ord_5"].str[0]

ddall["ord_5b"]=ddall["ord_5"].str[1]

drop_cols.append("ord_5")
for col in ["nom_5", "nom_6", "nom_7", "nom_8", "nom_9"]:

    train_vals = set(dd0[col].unique())

    test_vals = set(ddtest0[col].unique())

   

    xor_cat_vals=train_vals ^ test_vals

    if xor_cat_vals:

        ddall.loc[ddall[col].isin(xor_cat_vals), col]="xor"
X=ddall[ddall.columns.difference(["id", "target"] + drop_cols)]
X_oh=X[X.columns.difference(["ord_1", "ord_4", "ord_5a", "ord_5b", "day", "month"])]

oh1=pd.get_dummies(X_oh, columns=X_oh.columns, drop_first=True, sparse=True)

ohc1=oh1.sparse.to_coo()
from sklearn.base import TransformerMixin

from itertools import repeat

import scipy





class ThermometerEncoder(TransformerMixin):

    """

    Assumes all values are known at fit

    """

    def __init__(self, sort_key=None):

        self.sort_key = sort_key

        self.value_map_ = None

    

    def fit(self, X, y=None):

        self.value_map_ = {val: i for i, val in enumerate(sorted(X.unique(), key=self.sort_key))}

        return self

    

    def transform(self, X, y=None):

        values = X.map(self.value_map_)

        

        possible_values = sorted(self.value_map_.values())

        

        idx1 = []

        idx2 = []

        

        all_indices = np.arange(len(X))

        

        for idx, val in enumerate(possible_values[:-1]):

            new_idxs = all_indices[values > val]

            idx1.extend(new_idxs)

            idx2.extend(repeat(idx, len(new_idxs)))

            

        result = scipy.sparse.coo_matrix(([1] * len(idx1), (idx1, idx2)), shape=(len(X), len(possible_values)), dtype="int8")

            

        return result
thermos=[]

for col in ["ord_1", "ord_2", "ord_3", "ord_4", "ord_5a", "day", "month"]:

    if col=="ord_1":

        sort_key=['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster'].index

    elif col=="ord_2":

        sort_key=['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot'].index

    elif col in ["ord_3", "ord_4", "ord_5a"]:

        sort_key=str

    elif col in ["day", "month"]:

        sort_key=int

    else:

        raise ValueError(col)

    

    enc=ThermometerEncoder(sort_key=sort_key)

    thermos.append(enc.fit_transform(X[col]))
ohc=scipy.sparse.hstack([ohc1] + thermos).tocsr()

display(ohc)



X_train = ohc[:num_train]

X_test = ohc[num_train:]

y_train = dd0["target"].values
clf=LogisticRegression(C=0.123456789, solver="lbfgs", max_iter=5000)  # MODEL



clf.fit(X_train, y_train)



pred=clf.predict_proba(X_test)[:,1]



pd.DataFrame({"id": ddtest0["id"], "target": pred}).to_csv("submission.csv", index=False)
from sklearn.model_selection import cross_validate



score=cross_validate(clf, X_train, y_train, cv=3, scoring="roc_auc")["test_score"].mean()

print(f"{score:.6f}")