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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")
test = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_test.csv")
train.shape
train.head()
y = train["label"]
X = train.drop(["label"], axis=1)
y_test = test["label"]
test = test.drop("label", axis=1)
def show(d):
    d = np.asarray(d)
    d = d.reshape(28, 28)
    plt.imshow(d, cmap="binary")
    plt.axis("off")
    plt.show()
for i in range(3):
    show(X.iloc[i])
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X)
exp_var = pca.explained_variance_ratio_
cum_exp_var = np.cumsum(exp_var)
fig, ax = plt.subplots()
ax.plot(range(pca.n_components_), cum_exp_var)
ax.axhline(y=0.9, linestyle="--")
ax.set_xlabel("n_components")
ax.set_ylabel("cum exp variance")
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X)
test_pca = pca.transform(test)
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_pca, y)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, xgb.predict(test_pca))