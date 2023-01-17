# ライブラリをインポート

import numpy as np

import pandas as pd

from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix # 回帰問題における性能評価に関する関数

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.model_selection import train_test_split 

from sklearn.model_selection import KFold 

from sklearn.metrics import mean_absolute_error

import seaborn  as sns





# データの読込

cf_data = pd.read_csv("../input/skillup_ai/ks-projects-201801.csv", encoding="cp932")



#データの大きさと項目を確認

print(cf_data.shape)

print(cf_data.columns)



# データの表示

print(cf_data)

cf_data.head(50)

cf_data.describe()
#不要なデータを削除

cf_data_rem = ['goal', 'pledged', 'usd pledged', 'usd_pledged_real', 'ID', 'name', 'category', 'backers', 'country', 'Unnamed: 15', 'Unnamed: 16']

data_exp = cf_data.drop(cf_data_rem, axis = 1)

data_exp.head()
# dead_line から launched を差し引きし，period を導出  

data_exp['launched'] = pd.to_datetime(data_exp['launched'], errors = 'coerce') #エラー値はNaNに変換

data_exp['deadline'] = pd.to_datetime(data_exp['deadline'], errors = 'coerce') #エラー値はNaNに変換

data_exp['period'] = (data_exp['deadline'] - data_exp['launched']).dt.days



# deadlineとlaunchedを削除

data_exp = data_exp.drop(['deadline', 'launched'], axis=1)

data_exp.head()
# 欠損値（NULL）の確認

print(data_exp.isnull().any())

print(data_exp.isnull().sum(axis=0))

# 各列の統計量（平均，標準偏差など）の要約を取得

display(data_exp.describe)
states = data_exp["state"].unique()

categories = data_exp["main_category"].unique()

num_cat = len(categories)

print(states)

print(categories)

print(num_cat, "main_categoies")
# Success と Failed の抽出

data_exp = data_exp[(data_exp['state'] == 'successful') | (data_exp['state'] == 'failed')]

display(data_exp)

data_exp.describe(include='all')
#usd_goal_realの確認

sns.distplot(data_exp['usd_goal_real']);
#usd_goal_realにある程度幅が見られたため、対数をとる

data_exp['usd_goal_real'] = np.log(data_exp['usd_goal_real'] )

sns.distplot(data_exp['usd_goal_real']);
# One-Hotベクトル化

cat_onehot = pd.get_dummies(data_exp['main_category'])

data_exp = pd.concat([data_exp.drop(['main_category'], axis=1), cat_onehot], axis=1)

cur_onehot = pd.get_dummies(data_exp['currency'])

data_exp = pd.concat([data_exp.drop(['currency'], axis=1), cur_onehot], axis=1)

print(data_exp.shape)

# ホールドアウト法（テストデータを20%）

y = data_exp['state'].map({'failed' : 0, 'successful' : 1})

x = data_exp.drop('state', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

display(x_train.describe())

display(x_test.describe())

display(y_train.describe())

display(y_test.describe())
# ロジスティック回帰, 学習，結果を予測する

import matplotlib.pyplot as plt



linerR_data_exp = LogisticRegression()

linerR_data_exp.fit(x_train, y_train)

y_esti = linerR_data_exp.predict(x_test)

plt.hist(y_esti)

#split = 6

#split_num = 1



#result_df = pd.DataFrame( columns=['正答率（Accuracy）','適合率（Precision）','再現率（Recall）','F1値（F1-score）'] )



# テスト役を交代させながら学習と評価を繰り返す

#for train_idx, test_idx in KFold(n_splits=split, shuffle=True, random_state=1234).split(X, y):

 #   X_train, y_train = x_train[train_idx], y_train[train_idx] #学習用データ

 #  X_test, y_test = x_test[test_idx], y_test[test_idx]     #テスト用データ

    

    # 学習の実行

 #   clf = SGDClassifier(loss='log', penalty='none', max_iter=10000, fit_intercept=True,tol=1e-3)

 #   clf_std = clf.fit(X_train, y_train)

 #   y_pred_test = clf_std.predict(X_test)

    

    # 結果の計算、蓄積

 #   accuracy = accuracy_score(y_test, y_pred_test) # 正答率

 #  precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred_test) # 適合率・再現率・F1値

 #  tmp_se = pd.Series( [100 * accuracy, 100 * precision[1], 100 * recall[1], 100 * f1_score[1]], index=result_df.columns )

 #  result_df = result_df.append( tmp_se, ignore_index=True )



# 結果の表示

#result_df2 = pd.concat([result_df,pd.DataFrame(result_df.mean(axis=0),columns=['平均']).T])

#result_df2.head(100)
# 正解率，適合率，再現率を表示

print("正解率 =  :{:.2}".format(accuracy_score(y_test, y_esti)))

print("適合率 =  {:.2}".format(precision_score(y_test, y_esti)))

print("再現率 =  {:.2}".format(recall_score(y_test, y_esti)))

print("F1値 =  {:.2}".format(f1_score(y_test, y_esti)))