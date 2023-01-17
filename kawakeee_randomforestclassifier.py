# 初回のみ実行すればよい

!pip install mglearn
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import mglearn
from sklearn.datasets import make_moons

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



# データセットの用意

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,

                                                    random_state=42)

# ランダムフォレストの適用

forest = RandomForestClassifier(n_estimators=5, random_state=2)

forest.fit(X_train, y_train)
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):

    ax.set_title("Tree {}".format(i))

    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)

    

mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1],

                                alpha=.4)

axes[-1, -1].set_title("Random Forest")

mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()



X_train, X_test, y_train, y_test = train_test_split(

    cancer.data, cancer.target, random_state=0)

forest = RandomForestClassifier(n_estimators=100, random_state=0)

forest.fit(X_train, y_train)



print("訓練スコア: {:.3f}".format(forest.score(X_train, y_train)))

print("テストスコア:{:.3f}".format(forest.score(X_test, y_test)))
def plot_feature_importances_cancer(model):

    n_features = cancer.data.shape[1]

    plt.barh(np.arange(n_features), model.feature_importances_, align='center')

    plt.yticks(np.arange(n_features), cancer.feature_names)

    plt.xlabel("Feature importance")

    plt.ylabel("Feature")

    plt.ylim(-1, n_features)
plot_feature_importances_cancer(forest)