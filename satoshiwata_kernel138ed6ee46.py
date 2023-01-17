%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

from mpl_toolkits.mplot3d import Axes3D #3D散布図の描画

import seaborn as sns



import codecs as cd

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import log_loss, confusion_matrix

from sklearn.model_selection import train_test_split # ホールドアウト法に関する関数

from sklearn.model_selection import KFold # 交差検証法に関する関数

from sklearn.metrics import mean_absolute_error # 回帰問題における性能評価に関する関数
df = pd.read_csv("../1_data/ks-projects-201801.csv")

display(df.head())

df.describe()
#成功と失敗のみ抽出

df = df[(df['state'] == 'successful') | (df['state'] == 'failed')]
# 変数の削除

df = df.drop('usd_pledged_real', axis = 1)

df = df.drop('usd pledged', axis = 1)

df = df.drop('pledged', axis = 1)

df = df.drop('backers', axis = 1)

df = df.drop('ID', axis = 1)

df = df.drop('name', axis = 1)

df = df.drop('goal', axis = 1)

df = df.drop('category', axis = 1)

df.head()
#launched,deadlineを期間periodに置き換える

df['deadline'] = pd.to_datetime(df['deadline'], errors = 'coerce')

df['launched'] = pd.to_datetime(df['launched'], errors = 'coerce')

df['period'] = (df['deadline'] - df['launched']).dt.days

df = df.drop(['deadline', 'launched'], axis=1)

#histogram

sns.distplot(df['period']);

display(df.describe())
#ダミー変数に置き換え onehot

df = pd.get_dummies(df, columns = ["main_category", "country", "currency"])

df
#散布図行列を書いてみる

pd.plotting.scatter_matrix(df,figsize=(10,10))

plt.show()
# 相関係数を確認

df.corr()



# 相関係数をヒートマップにして可視化

sns.heatmap(df.corr())

plt.show()
#目的変数と説明変数

y = df['state']

X = df.drop('state', axis=1)



# ロジスティック回帰

clf = SGDClassifier(loss='log', penalty='none', max_iter=10000, fit_intercept=True, random_state=1234, tol=1e-3)

clf.fit(X, y)



# ラベルを予測

y_est = clf.predict(X)



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
# ここからDAY2

# 欠損値の確認　欠損値がある行数の確認

display(df.isnull().sum())
# データのスケール

# usd_goal_realについて

sns.distplot(df['usd_goal_real']);
#目的変数と説明変数

y = data2['state'].map({'successful': 1, 'failed': 0})

X = df.drop('state', axis=1)

#ホールドアウト法  test用20%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 







# trainを標準化、testはtrainの値を用いて標準化する

# usd_goal_realを標準化する

stdsc = StandardScaler()

stdsc.fit_transform(df2[["usd_goal_real"]].values)



# periodを標準化する

stdsc.fit_transform(df2[["period"]].values)