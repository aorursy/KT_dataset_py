%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

from sklearn.metrics import mean_squared_error, mean_absolute_error # 回帰問題における性能評価に関する関数

from  sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix # 回帰問題における性能評価に関する関数



import seaborn as sns



import datetime # 日付処理用
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/kickstarter-projects/ks-projects-201801.csv', delimiter=",") #読み込んだCSVファイルを” , ”で区切ります
df.head(10)
df.columns #データの列を確認
df['state'].unique() #stateの詳細確認
df = df[df['state'].isin(['failed', 'successful'])]

df['state'].unique()
df.info() #データの情報を確認
df.isnull().sum() #データの欠損値を確認
df.corr() #データの相関係数を確認
sns.heatmap(df.corr())

plt.show() #ヒートマップによる可視化
#利用しないカラムの削除

df = df.drop(['ID','name','category','country'], axis=1)
df.head(10) #再度データの確認
#目的変数と説明変数

y = df['state'] =='successful'

X = df[["usd_pledged_real", "backers"]].values



#学習

from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(loss='log', penalty='none', max_iter=10000, fit_intercept=True, random_state=1234, tol=1e-3)

clf.fit(X, y)



# ラベルを予測

y_est = clf.predict(X)
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, confusion_matrix



# 対数尤度を表示

display('対数尤度 = {:.3f}'.format(- log_loss(y, y_est)))



# 正答率accuracy, 適合率precision, 再現率recallを表示

display('正答率 = {:.3f}%'.format(100 * accuracy_score(y, y_est)))

display('適合率 = {:.3f}%'.format(100 * precision_score(y, y_est)))

display('再現率 = {:.3f}%'.format(100 * recall_score(y, y_est)))



# 予測値と正解のクロス集計

conf_mat = pd.DataFrame(confusion_matrix(y, y_est), 

                        index=['actual = others', 'actual = successful'], 

                        columns=['predict = others', 'predict = successful'])

display(conf_mat)