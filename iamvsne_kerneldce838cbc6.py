%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, precision_score, recall_score

from sklearn.svm import SVC
df_ks = pd.read_csv("../input/ks-projects-201801.csv")

display(df_ks.head())

df_ks.describe()
#計算が重いのでランダム5000サンプル、4000トレーニング、1000テスト

df = df_ks.sample(n=5000, random_state=10)



#使える変数のみのデータセットにする

drop_col = ['pledged','backers','usd pledged','usd_pledged_real','usd_goal_real']

df = df.drop(drop_col, axis=1)



#nanを削除

df = df.dropna()



#開始から締め切り日までのUNIX時間（秒）を計算して追加

df['period'] = pd.to_datetime(df['deadline']).map(pd.Timestamp.timestamp) - pd.to_datetime(df['launched']).map(pd.Timestamp.timestamp)

display(df.head())

df_train  = df[:4000]

display(df_train.head())

df_test = df[4000:]

display(df_test.head())
#目標金額のヒストグラム表示。キリのいい値段のとき失敗率が上がっていることがわかる。

sccss = df_train[df_train['state']=="successful"]['goal'].values

othrs = df_train[df_train['state']!="successful"]['goal'].values



plt.title('distribution of goal')

plt.hist([x for x in sccss if 0<x and x<10000],bins=100,color="#5F9BFF", alpha=.5,range=(0,10000))

plt.hist([x for x in othrs if 0<x and x<10000],bins=100,color="#F8766D", alpha=.5,range=(0,10000))



plt.xlabel("goal")

plt.ylabel("fleq")



plt.show()
#正解ラベルの付与

#display(df_train)

df_train['label'] = 0

df_test['label'] = 0

df_train = df_train.reset_index()

df_test = df_test.reset_index()

#print(df_train['state'])



#display(df_main['label'])

#stateがsuccessfulのときは1,そうでないときは-1のラベルを付ける

for i in range(len(df_train)):

    if df_train['state'][i] == 'successful':

        df_train['label'][i] = True

    else:

        df_train['label'][i] = False

display(df_train)



for i in range(len(df_test)):

    if df_test['state'][i] == 'successful':

        df_test['label'][i] = True

    else:

        df_test['label'][i] = False

display(df_test)
#goal金額が500ドルの倍数のとき1、そうでないとき0

df_train['multiple_500'] = 0

df_test['multiple_500'] = 0



for i in range(len(df_train)):

    if df_train['goal'][i]%500 == 0:

        df_train['multiple_500'][i] = 1

        

for i in range(len(df_test)):

    if df_test['goal'][i]%500 == 0:

        df_test['multiple_500'][i] = 1
#単語のカウント



word_count_suc = {}

word_count_otr = {}

for i in range(len(df_train)):

    if df_train['state'][i] == 'successful':

        for word in df_train['name'][i].split():

            if not word in word_count_suc:

                word_count_suc[word] = 0

            word_count_suc[word] += 1

#print(word_count_train)



for i in range(len(df_train)):

    if df_train['state'][i] != 'successful':

        for word in df_train['name'][i].split():

            if not word in word_count_otr:

                word_count_otr[word] = 0

            word_count_otr[word] += 1



#出現回数順にソート

count_sorted_suc = sorted(word_count_suc.items(), key=lambda x:x[1], reverse=True)

count_sorted_otr = sorted(word_count_otr.items(), key=lambda x:x[1], reverse=True)



#print(count_sorted_suc)



#成功プロジェクトに多い単語、そうでないプロジェクトに多い単語を抽出

#どちらにも出てくる単語は除外

successful_words = list(set([count_sorted_suc[i][0] for i in range(200)]) - set([count_sorted_otr[i][0] for i in range(300)]))

unsuccessful_words = list(set([count_sorted_otr[i][0] for i in range(200)]) - set([count_sorted_suc[i][0] for i in range(300)]))



#適切でなさそうな単語を除外

successful_words.remove('2012')

successful_words.remove('2014')

successful_words.remove('2017')

unsuccessful_words.remove('(Canceled)')

unsuccessful_words.remove('(Suspended)')

print(successful_words)

print(unsuccessful_words)

#上記成功率に関わりそうな単語をまとめる

words = successful_words

words.extend(unsuccessful_words)



#上記単語が出現したら1、そうでないときは0

for word in words:

    df_train[word] = 0

    df_test[word] = 0

    for i in range(len(df_train)):

        for word_in_name in df_train['name'][i].split():

            if word_in_name == word:

                df_train[word][i] = 1

    for j in range(len(df_test)):

        for word_in_name in df_test['name'][j].split():

            if word_in_name == word:

                df_test[word][j] = 1

display(df_test)
#goal金額、開始日から終了日までの期間、設定金額が500ドルの倍数か、頻出単語に引っかかるか　を説明変数とする

valiables = words

valiables.extend(['goal','period','multiple_500'])



y_train = df_train["label"].values

#print(y_train)

for i in range(len(y_train)):

    if y_train[i] == True:

        y_train[i] = 1

    else:

        y_train[i] = 0

y = y_train.astype('int')

X = df_train[valiables].values

clf = SGDClassifier(loss='log', penalty='none', max_iter=10000, fit_intercept=True, random_state=1234)

clf.fit(X, y)

#ロジスティック回帰

X_test = df_test[valiables].values

y_test_tmp = df_test['label'].values

y_test = y_test_tmp

for i in range(len(y_test_tmp)):

    if y_test[i] == True:

        y_test[i] = 1

    else:

        y_test[i] = 0

y_test = y_test.astype('int')



# ラベルを予測

y_est = clf.predict(X_test)



# 対数尤度を表示

print('対数尤度 = {:.3f}'.format(- log_loss(y_test, y_est)))



# 正答率を表示

print('正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_est)))
#SVM

#y_train_svm = y_train

#for i in range(len(y_train)):

#    if y_train[i] == True:

#        y_train_svm[i] = 1

#    else:

#        y_train_svm[i] = -1

#y_train_svm = y_train_svm.astype('int')

#print(y_train_svm)

#print(np.shape(X))



#C = 5

#clf = SVC(C=C, kernel="linear")

#clf.fit(X, y_train_svm)