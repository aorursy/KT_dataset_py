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
%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, confusion_matrix



# loading the data

df = pd.read_csv('../input/kickstarter-projects/ks-projects-201801.csv', parse_dates=['deadline', 'launched'])



# knowning the main informations of the data

display('dataframeの行数・列数の確認')

display(df.shape)



display('columnの確認')

display(df.columns)



display('dataframeの各列のデータ型を確認')

display(df.dtypes)



# 先頭5行を表示

display(df.head())



# # '*****'にはどんなデータが入っているか確認

# print(df['main_category'].unique())

# print(df['category'].unique())



# categoryとmain_categoryのindexを区別

df['category'] = 'sc_' + df['category']

df['main_category'] = 'mc_' + df['main_category']





# # 要約統計量の表示

# df.describe()



# 欠損値の確認

display(df.isnull().sum(axis = 0))



# stateの種類を確認

df.groupby('state')['ID'].count()
# データの区分け

# 共通

df_0 = df[['ID', 'name']]

display(df_0.head())



# 事前にわかるデータ

df_1 = df[['category', 'main_category', 'currency', 'deadline', 'launched', 'country', 'usd_goal_real']]

display(df_1.head())



# 事前にはわからないデータ

df_2 = df[['state', 'backers', 'usd_pledged_real']]

display(df_2.head())



# 不要データ

df_3 = df[['goal', 'pledged', 'usd pledged']]

display(df_3.head())

# '*****'にはどんなデータが入っているか確認

# display(df_1['category'].unique())

# display(df_1['main_category'].unique())

# display(df_1['currency'].unique())

# display(df_1['country'].unique())





onehot_category = pd.get_dummies(df_1['category'])

display(onehot_category.head())



onehot_main_category = pd.get_dummies(df_1['main_category'])

display(onehot_main_category.head())



onehot_currency = pd.get_dummies(df_1['currency'])

display(onehot_currency.head())



onehot_country = pd.get_dummies(df_1['country'])

display(onehot_country.head())
# launchedの整理

# display(df_1['launched'].head())

df_1['launched_year']   = df_1['launched'].dt.year

df_1['launched_month']   = df_1['launched'].dt.month

df_1['launched_day']   = df_1['launched'].dt.day



# deadlineの整理

# display(df_1['deadline'].head())

df_1['deadline_year']   = df_1['deadline'].dt.year

df_1['deadline_month']   = df_1['deadline'].dt.month

df_1['deadline_day']   = df_1['deadline'].dt.day



display(df_1.head())
# 説明変数の設定

X = onehot_main_category.join([onehot_category, onehot_currency, onehot_country, df_1['usd_goal_real'],

                          df_1['launched_year'], df_1['launched_month'], df_1['launched_day'],

                          df_1['deadline_year'], df_1['deadline_month'], df_1['deadline_day']])

# display(X.head(10))



# 目的関数の設定

y = df_2['state'] == 'successful'

y = y*1

# display(y.head(10))

# データの図示

check = pd.concat([y, X['usd_goal_real'], X['launched_year'], X['launched_month'], X['launched_day'],

                   X['deadline_year'], X['deadline_month'], X['deadline_day']], axis=1)

# display(check.head())





# 散布図行列を書いてみる

pd.plotting.scatter_matrix(check, figsize=(10,10))

plt.show()



# 相関係数を確認

check.corr()



# 相関係数をヒートマップにして可視化

sns.heatmap(check.corr())

plt.show()
# ---------------------------------------------------------------

# ロジスティック回帰

clf = SGDClassifier(loss='log', penalty='none', max_iter=10000, fit_intercept=True, random_state=1234, tol=1e-3)

clf.fit(X, y)



# ラベルを予測

y_est = clf.predict(X)
# display(sum(y))

# display(sum(y_est))



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