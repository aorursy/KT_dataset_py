# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
mushrooms = pd.read_csv('../input/mushrooms.csv')
mushrooms.describe()
from sklearn.base import TransformerMixin
from sklearn.preprocessing import label_binarize

class LabelToRatio(TransformerMixin):
    
    def __init__(self):
        self._ratios = []
    
    def fit(self, X, y):
        y_binalized = label_binarize(y, np.unique(y)).ravel()
        for i in range(X.shape[1]):
            ratio = pd.DataFrame({"x": X[:, i],
                                  "y": y_binalized}) \
                .groupby("x") \
                .mean() \
                .reset_index()
            self._ratios.append(ratio)

    def transform(self, X, y=None):
        X_ratio = np.zeros_like(X)
        for i in range(X.shape[1]):
            X_i = pd.DataFrame({"x": X[:, i]}) \
                .merge(self._ratios[i], on="x", how="left") \
                .fillna(0) \
                .loc[:, "y"] \
                .values
            X_ratio[:, i] = X_i
        return X_ratio
X = mushrooms.iloc[:, 1:].values
y = mushrooms.iloc[:, 0].values
label_to_ratio = LabelToRatio()
label_to_ratio.fit(X,y)
X_ratio = label_to_ratio.transform(X)
X_ratio[:5, :5]
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_ratio)
y_binarized = label_binarize(y, np.unique(y))
mushroom_pca = pd.DataFrame(np.hstack((X_pca, y_binarized)))
mushroom_pca.columns = ["c0", "c1", "class"]

import plotly.offline as offline
import plotly.graph_objs as go

offline.init_notebook_mode()

index_0 = mushroom_pca["class"] == 0
index_1 = mushroom_pca["class"] == 1
trace0 = go.Scatter(x=mushroom_pca.loc[index_0, "c0"],
                    y=mushroom_pca.loc[index_0, "c1"],
                    name="0",
                    mode="markers")
trace1 = go.Scatter(x=mushroom_pca.loc[index_1, "c0"],
                    y=mushroom_pca.loc[index_1, "c1"],
                    name="1",
                    mode="markers")                    
offline.iplot([trace0, trace1])
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(mushrooms.iloc[:, 1:].values,
                                                    mushrooms.iloc[:, 0].values)
label_to_ratio = LabelToRatio()
label_to_ratio.fit(X_train, y_train)
X_train, X_test = label_to_ratio.transform(X_train), label_to_ratio.transform(X_test)
svc = SVC(probability=True)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print(classification_report(y_test, y_pred))
import scikitplot as skplt
y_proba = svc.predict_proba(X_test)
skplt.metrics.plot_roc_curve(y_test, y_proba)
