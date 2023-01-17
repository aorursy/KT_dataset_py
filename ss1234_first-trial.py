# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#データを読み込む

import pandas as pd

data=pd.read_csv('../input/data.csv')

data.head()
#columnsの詳細説明。どのくらい揃っているのか、データ型などが記してある

data.info()
col = data.columns       # .columns gives columns names in data 

print(col)
y = data.diagnosis # M or B 

list = ['Unnamed: 32','id','diagnosis'] #data.info()で見たcolumnsのうち絶対いらないのを除く

X= data.drop(list,axis = 1 ) #axis=1は

X.head()
#単純変量統計という特徴量選択を行う

from sklearn.feature_selection import SelectPercentile, f_classif #MまたはBという分類問題なのでclassification

selector = SelectPercentile(score_func=f_classif, percentile=40) 



selector.fit(X, y) #SelectPercentile関数にfitさせる

mask = selector.get_support() #SelectPercentile関数にfitさせたあとの選択結果関数

print(col) #選択する前の全特徴量

print(mask) #上記の特徴量のうちどれが選択され、されなかったかを表示



# 選択した特徴量の列のみ取得

X_selected = selector.transform(X) #XをSelectPercentile関数にfitさせて変換する

print("X.shape={}, X_selected.shape={}".format(X.shape, X_selected.shape))
#訓練データとテストデータに分けた

from sklearn.model_selection import train_test_split

X_selected_train, X_selected_test, y_train, y_test=train_test_split(X_selected, y, random_state=42) #test_sizeの初期値=0,25
#教師あり学習/分類/2クラス分類/勾配ブースティング回帰木

from sklearn.ensemble import GradientBoostingClassifier



gbrt=GradientBoostingClassifier(random_state=42)

gbrt.fit(X_selected_train, y_train)



#評価する

print("Accuracy on training set:{:.3f}".format(gbrt.score(X_selected_train, y_train)))

print("Accuracy on training set:{:.3f}".format(gbrt.score(X_selected_test, y_test)))
#深さの最大値を制限してより強力な事前枝刈りを行うか、学習率を下げよう

gbrt=GradientBoostingClassifier(random_state=42, learning_rate=0.07)

gbrt.fit(X_selected_train, y_train)



#再学習して再評価

print("Accuracy on training set:{:.3f}".format(gbrt.score(X_selected_train, y_train)))

print("Acccuracy on trainig set:{:.3f}".format(gbrt.score(X_selected_test, y_test)))
#勾配ブースティング回帰木での最高スコア=0.958
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



def plot_feature_importances_col(model):

    n_features=X_selected.shape[1]

    plt.barh(range(n_features), model.feature_importances_, align='center')

    plt.yticks(np.arange(n_features), col)

    plt.xlabel("Feature importances")

    plt.ylabel("Feature")



plot_feature_importances_col(gbrt)