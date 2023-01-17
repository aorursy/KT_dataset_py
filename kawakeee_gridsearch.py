# 初回のみ実行すればよい

!pip install mglearn
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import mglearn

from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris
from sklearn.svm import SVC



iris = load_iris()



X_train, X_test, y_train, y_test = train_test_split(

    iris.data, iris.target, random_state=0)

print("訓練セットサイズ: {}   テストセットサイズ: {}".format(

      X_train.shape[0], X_test.shape[0]))



best_score = 0



for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:

    for C in [0.001, 0.01, 0.1, 1, 10, 100]:

        # それぞれのパラメータの組合せに対して学習モデルを訓練させる

        svm = SVC(gamma=gamma, C=C)

        svm.fit(X_train, y_train)

        # 訓練後にテストスコアを算出する

        score = svm.score(X_test, y_test)

        # スコアが高ければ最良スコアとして保持する

        if score > best_score:

            best_score = score

            best_parameters = {'C': C, 'gamma': gamma}



print("最良スコア: {:.2f}".format(best_score))

print("最良パラメータ: {}".format(best_parameters))
mglearn.plots.plot_threefold_split()
from sklearn.svm import SVC

# （訓練データ+検証データ）と（テストデータ）にデータセットを分割する

X_trainval, X_test, y_trainval, y_test = train_test_split(

    iris.data, iris.target, random_state=0)

# さらにもう一度（訓練データ+検証データ）を、（訓練データ）と（検証データ）に分割する

X_train, X_valid, y_train, y_valid = train_test_split(

    X_trainval, y_trainval, random_state=1)

print("訓練セットサイズ: {}   検証セットサイズ: {}   テストセットサイズ:"

      " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))



best_score = 0



for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:

    for C in [0.001, 0.01, 0.1, 1, 10, 100]:

        # それぞれのパラメータの組合せに対して学習モデルを訓練させる

        svm = SVC(gamma=gamma, C=C)

        svm.fit(X_train, y_train)

        # 訓練後に、検証データセットを用いてスコアを算出する

        score = svm.score(X_valid, y_valid)

        # スコアが高ければ最良スコアとして保持する

        if score > best_score:

            best_score = score

            best_parameters = {'C': C, 'gamma': gamma}



# 最良パラメータの学習モデルを生成する

svm = SVC(**best_parameters)

# 訓練データ+検証データでモデルを再構築する

svm.fit(X_trainval, y_trainval)

test_score = svm.score(X_test, y_test)

print("最良検証スコア: {:.2f}".format(best_score))

print("最良パラメータ: ", best_parameters)

print("テストスコア: {:.2f}".format(test_score))
from sklearn.model_selection import cross_val_score

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:

    for C in [0.001, 0.01, 0.1, 1, 10, 100]:

        # それぞれのパラメータの組合せに対して学習モデルを訓練させる

        svm = SVC(gamma=gamma, C=C)

        # 分割数5で層化分割交差検証のスコアを取得する

        scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)

        # 交差検証スコアの平均値を算出する

        score = np.mean(scores)

        # iスコアが高ければ最良スコアとして保持する

        if score > best_score:

            best_score = score

            best_parameters = {'C': C, 'gamma': gamma}

# 最良パラメータの学習モデルを生成する

svm = SVC(**best_parameters)

# 訓練データ+検証データでモデルを再構築する

svm.fit(X_trainval, y_trainval)
mglearn.plots.plot_cross_val_selection()
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],

              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

print("パラメータグリッド:\n{}".format(param_grid))
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

grid_search = GridSearchCV(SVC(), param_grid, cv=5)
X_train, X_test, y_train, y_test = train_test_split(

    iris.data, iris.target, random_state=0)
grid_search.fit(X_train, y_train)
print("テストセットスコア: {:.2f}".format(grid_search.score(X_test, y_test)))
print("最良パラメータ: {}".format(grid_search.best_params_))

print("最良交差検証スコア: {:.2f}".format(grid_search.best_score_))
# Dataframeに変換

results = pd.DataFrame(grid_search.cv_results_)

# 最初の5行を表示

display(results.head())
scores = np.array(results.mean_test_score).reshape(6, 6)



# 平均検証スコアのヒートマップ

mglearn.tools.heatmap(scores, xlabel='gamma', xticklabels=param_grid['gamma'],

                      ylabel='C', yticklabels=param_grid['C'], cmap="viridis")
fig, axes = plt.subplots(1, 3, figsize=(13, 5))



param_grid_linear = {'C': np.linspace(1, 2, 6),

                     'gamma':  np.linspace(1, 2, 6)}



param_grid_one_log = {'C': np.linspace(1, 2, 6),

                      'gamma':  np.logspace(-3, 2, 6)}



param_grid_range = {'C': np.logspace(-3, 2, 6),

                    'gamma':  np.logspace(-7, -2, 6)}



for param_grid, ax in zip([param_grid_linear, param_grid_one_log,

                           param_grid_range], axes):

    grid_search = GridSearchCV(SVC(), param_grid, cv=5)

    grid_search.fit(X_train, y_train)

    scores = grid_search.cv_results_['mean_test_score'].reshape(6, 6)



    # 平均交差検証をプロット

    scores_image = mglearn.tools.heatmap(

        scores, xlabel='gamma', ylabel='C', xticklabels=param_grid['gamma'],

        yticklabels=param_grid['C'], cmap="viridis", ax=ax)



plt.colorbar(scores_image, ax=axes.tolist())