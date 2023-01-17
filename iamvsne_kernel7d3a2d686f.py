%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, precision_score, recall_score
df_ks = pd.read_csv("../input/ks-projects-201801.csv")

display(df_ks.head())

df_ks.describe()

#print(pd.get_dummies(df_ks['state'])['successful'])
sccss = df_ks[df_ks['state']=="successful"]['backers'].values

othrs = df_ks[df_ks['state']!="successful"]['backers'].values



plt.hist([x for x in sccss if 0<x and x<200],bins=100,color="#5F9BFF", alpha=.5)

plt.hist([x for x in othrs if 0<x and x<200],bins=100,color="#F8766D", alpha=.5)



plt.xlabel("backers")

plt.ylabel("fleq")



plt.show()
sccss = df_ks[df_ks['state']=="successful"]['pledged'].values

othrs = df_ks[df_ks['state']!="successful"]['pledged'].values



plt.hist([x for x in sccss if 0<x and x<10000],bins=100,color="#5F9BFF", alpha=.5)

plt.hist([x for x in othrs if 0<x and x<10000],bins=100,color="#F8766D", alpha=.5)



plt.xlabel("pledged")

plt.ylabel("fleq")



plt.show()

#pledgeが多いほど成功する？
sccss = df_ks[df_ks['state']=="successful"]['main_category'].values

othrs = df_ks[df_ks['state']!="successful"]['main_category'].values

plt.hist(sccss,color="#5F9BFF", alpha=.5)

plt.hist(othrs,color="#5F9BFF", alpha=.5)

plt.xlabel("main_category")

plt.ylabel("fleq")

plt.show()

#kaggle上だとこのセルではエラーが出る
sccss = df_ks[df_ks['state']=="successful"]['goal'].values

othrs = df_ks[df_ks['state']!="successful"]['goal'].values



plt.hist([x for x in sccss if 0<x and x<10000],bins=100,color="#5F9BFF", alpha=.5)

plt.hist([x for x in othrs if 0<x and x<10000],bins=100,color="#F8766D", alpha=.5)



plt.xlabel("goal")

plt.ylabel("fleq")



plt.show()
sccss = df_ks[df_ks['state']=="successful"]['country'].values

othrs = df_ks[df_ks['state']!="successful"]['country'].values

plt.hist(sccss,color="#5F9BFF", alpha=.5)

plt.hist(othrs,color="#F8766D", alpha=.5)

plt.xlabel("country")

plt.ylabel("fleq")

plt.show()

#kaggle上だとこのセルではエラーが出る
#計算が重いため上から1000個

df_ks = df_ks[:1000]



#一列目がpladged、二列目がbackersの配列を作る（successfulが1のラベル、それ以外が0のラベル）

data2 = np.stack([df_ks[df_ks['state']=="successful"]['pledged'].values,df_ks[df_ks['state']=="successful"]['backers'].values], axis=1) 

label1 =  np.ones(len(data2))



data1 = np.stack([df_ks[df_ks['state']!="successful"]['pledged'].values,df_ks[df_ks['state']!="successful"]['backers'].values], axis=1) 

label0 = np.zeros(len(data1))
#プロット

plt.grid(which='major',color='black',linestyle=':')

plt.grid(which='minor',color='black',linestyle=':')

plt.plot(data2[:, 0], data2[:, 1], 'o', color='C0', label='success')

plt.plot(data1[:, 0], data1[:, 1], '^', color='C1', label='others')

plt.legend(loc='best')

plt.show()
X = np.concatenate([data2, data1])

y = np.concatenate([label0, label1])

clf = SGDClassifier(loss='log', penalty='none', max_iter=10000, fit_intercept=True,random_state=1234)

clf.fit(X, y)



# 重みを取得して表示

w0 = clf.intercept_[0]

w1 = clf.coef_[0, 0]

w2 = clf.coef_[0, 1]

print("w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}".format(w0, w1, w2))
# データをプロット

plt.grid(which='major',color='black',linestyle=':')

plt.grid(which='minor',color='black',linestyle=':')

plt.plot(data2[:, 0], data2[:, 1], 'o', color='C0', label='success')

plt.plot(data1[:, 0], data1[:, 1], '^', color='C1', label='others')



# 境界線をプロットして表示

# 紫：境界線

x1, x2 = X[:, 0], X[:, 1]

line_x = np.arange(np.min(x1) - 1, np.max(x1) + 1)

line_y = - line_x * w1 / w2 - w0 / w2

plt.plot(line_x, line_y, linestyle='-.', linewidth=3, color='purple', label='Threshold')

plt.ylim([np.min(x2) - 1, np.max(x2) + 1])

plt.legend(loc='best')

plt.xlim(-10000,100000)

plt.ylim(-100,1000)

plt.show()
# ラベルを予測

y_est = clf.predict(X)



# 対数尤度を表示

print('対数尤度 = {:.3f}'.format(-log_loss(y, y_est)))



# 正答率を表示

print('正答率 = {}%'.format(100 * accuracy_score(y, y_est)))
# 予測値と正解のクロス集計

conf_mat = pd.DataFrame(confusion_matrix(y, y_est), index=['予測値 = 1', '予測値 = 0'], columns=['正解 = 1', '正解 = 0'])

conf_mat
print(accuracy_score(y,y_est))

print(precision_score(y,y_est))

print(recall_score(y,y_est))