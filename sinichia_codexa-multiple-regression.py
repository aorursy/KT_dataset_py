# ライブラリのインポート

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# import seaborn as sns

# %matplotlib inline



# sns.set()



df = pd.read_csv("../input/007-022.csv")
# df.head()
df.iloc[0,0] = 64

df = df.rename(columns={"Pysics": "Physics"})
# df.loc[:, "Physics"]
X = np.array(df.iloc[:, :3])

y = np.array(df.iloc[:, 3])
# m = len(y)
# 点数をプロットしてみる

fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(df.loc[:, "Physics"], df.loc[:, "Science"], df.loc[:, "Math"], color="#ef1234")

ax.set_xlabel("Physics")

ax.set_ylabel("Science")

ax.set_zlabel("Math")

plt.show()
# X.shape[1]
# 正規化(Z-score Normalization)

def normalization(X):

    X_norm = np.zeros((X.shape[0], X.shape[1]))

    mean = np.zeros((1, X.shape[1]))

    std = np.zeros((1, X.shape[1]))

    for i in range(X.shape[1]):

        mean[:, i] = np.mean(X[:, i])

        std[:, i]  = np.std( X[:, i])

        X_norm[:, i] = (X[:, i] - mean[:, i]) / std[:, i]

    return X_norm, mean, std
X_norm, mean, std = normalization(X)
# X_norm[0:5, :]
# X[0:5, :]
# X_norm.std()
# 特徴量の前処理

X_padded = np.column_stack((np.ones((m, 1)), X_norm))
# X_padded[0:10, :]
# コスト関数

def cost(X, y, weight):

    m = len(y)

    J = 0

    y_hat = X.dot(weight)

    diff = (np.transpose([y]) - y_hat)**2

    J = 1 / (2*m) * diff.sum(axis = 0)

    return J
# コスト関数　ベクトル化

def cost_2(X, y, weight):

    m = len(y)

    J = 0

    y_shaped = y.reshape(m, 1)

    y_hat = X.dot(weight)

    J = 1 / (2*m) * (y_shaped - y_hat).T.dot(y_shaped - y_hat)

    return J
# weight_test = np.array([[20], [10], [5], [-1]])

# cost_2(X_padded, y, weight_test) == cost(X_padded, y, weight_test)
# 学習率と学習回数（イテレーション）

alpha = 0.01

num_iters = 500
# 最急降下法

def gradient_descent(X, y, weight, alpha, iterations):

    m = len(y)

    J_history = np.zeros((iterations, 1))

    

    for i in range(iterations):

        weight = weight - alpha / m * X.T.dot(X.dot(weight) - np.transpose([y]))

        J_history[i] = cost(X, y, weight)

    return weight, J_history
# パラメータ列ベクトルの初期化

weight_init = np.zeros((4,1))
# (X_padded.dot(weight_init) - y.T)[0].shape
# y.reshape(m,1).shape
weight, J_history = gradient_descent(X_padded, y, weight_init, alpha, num_iters)
# print(weight)
# J_history[0:10]
# コストと学習回数のグラフ

plt.plot(range(J_history.size), J_history, "-b", linewidth=1)

plt.xlabel("Number of iterations")

plt.ylabel("Cost J")

plt.grid(True)

plt.show()
# print(weight)
# コデクサ君 = 物理76点、化学96点、統計82点

# コデクサ君の数学の点数を予測したい

physics_norm = float((76 - mean[:, 0]) / std[:, 0])

science_norm = float((96 - mean[:, 1]) / std[:, 1])

statistics_norm = float((82 - mean[:, 2]) / std[:, 2])
# X0を追加し、特徴量X行列の作成

pred_padded = np.array([1, physics_norm, science_norm, statistics_norm])
# コデクサ君の数学の点数の予測

pred = float(pred_padded.dot(weight))
pred