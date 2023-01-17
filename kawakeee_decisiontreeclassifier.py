# 初回のみ実行すればよい

!pip install mglearn
import numpy as np

import pandas as pd

import mglearn
# 動物を区別する決定木

mglearn.plots.plot_animal_tree()
mglearn.plots.plot_tree_progressive()
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier



cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(

    cancer.data, cancer.target, stratify=cancer.target, random_state=42)

tree = DecisionTreeClassifier(random_state=0)

tree.fit(X_train, y_train)

print("訓練スコア: {:.3f}".format(tree.score(X_train, y_train)))

print("テストスコア:{:.3f}".format(tree.score(X_test, y_test)))
from sklearn.tree import export_graphviz

export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"],

                feature_names=cancer.feature_names, impurity=False, filled=True)

# 全体を確認

from IPython.display import Image, display_png

!dot -Tpng tree.dot -o tree.png

display_png(Image("tree.png"))
tree = DecisionTreeClassifier(max_depth=4, random_state=0)

tree.fit(X_train, y_train)



print("訓練スコア: {:.3f}".format(tree.score(X_train, y_train)))

print("テストスコア:{:.3f}".format(tree.score(X_test, y_test)))
from sklearn.tree import export_graphviz

export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"],

                feature_names=cancer.feature_names, impurity=False, filled=True)

# 全体を確認(木の深さが4であることを確認できる)

from IPython.display import Image, display_png

!dot -Tpng tree.dot -o tree.png

display_png(Image("tree.png"))
print("特徴量重要度の配列:")

print(tree.feature_importances_)
import matplotlib.pyplot as plt

def plot_feature_importances_cancer(model):

    n_features = cancer.data.shape[1]

    plt.barh(np.arange(n_features), model.feature_importances_, align='center')

    plt.yticks(np.arange(n_features), cancer.feature_names)

    plt.xlabel("Feature importance")

    plt.ylabel("Feature")

    plt.ylim(-1, n_features)



plot_feature_importances_cancer(tree)
tree = mglearn.plots.plot_tree_not_monotone()

display(tree)
import os

ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))
# semilogy: y軸を対数変換して散布図を描写

plt.semilogy(ram_prices.date, ram_prices.price)

plt.xlabel("Year")

plt.ylabel("Price in $/Mbyte")
from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

# 2000年を境に、訓練データ・テストデータに分割

data_train = ram_prices[ram_prices.date < 2000]

data_test = ram_prices[ram_prices.date >= 2000]



# 訓練データの説明変数を取得

X_train = data_train.date[:, np.newaxis]

# 訓練データの目的変数を対数変換して取得

y_train = np.log(data_train.price)



# `DecisionTreeRegressor`のモデルを構築

tree = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)

# `LinearRegression`のモデルを構築

linear_reg = LinearRegression().fit(X_train, y_train)



# ram_pricesの全日付を取得

X_all = ram_prices.date[:, np.newaxis]



# 全日付に対しての予測を行う

pred_tree = tree.predict(X_all)

pred_lr = linear_reg.predict(X_all)



# 各モデルの予測結果を取得

price_tree = np.exp(pred_tree)

price_lr = np.exp(pred_lr)
plt.semilogy(data_train.date, data_train.price, label="Training data")

plt.semilogy(data_test.date, data_test.price, label="Test data")

plt.semilogy(ram_prices.date, price_tree, label="Tree prediction")

plt.semilogy(ram_prices.date, price_lr, label="Linear prediction")

plt.legend()