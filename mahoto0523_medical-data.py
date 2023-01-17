# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pandas as pd
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/insurance.csv")
data.head()
print(data['sex'].unique())
print(data['region'].unique())
print(data['smoker'].unique())
sex_cat = data['sex'].unique()
sex_dict = {sex_cat[idx]: idx for idx in range(len(sex_cat))}
print(sex_dict)
reg_cat = data['region'].unique()
reg_dict = {reg_cat[idx]: idx for idx in range(len(reg_cat))}
print(reg_dict)
smk_cat = data['smoker'].unique()
smk_dict = {smk_cat[idx]: idx for idx in range(len(smk_cat))}
print(smk_dict)
for key in sex_dict.keys():
    data.loc[data.sex == key, 'sex'] = sex_dict[key]
for key in reg_dict.keys():
    data.loc[data.region == key, 'region'] = reg_dict[key]
for key in smk_dict.keys():
    data.loc[data.smoker == key, 'smoker'] = smk_dict[key]
data.head()
import matplotlib.pyplot as plt
import seaborn as sns

corr = data.corr()
print(corr)
sns.pairplot(data, size=2.5)
plt.show()
import numpy as np

# ages = [idx*10 for idx in range(1, 7)]
# means = {val: np.mean(data.loc[(data.age >= val) & (data.age < val+10), 'charges']) for val in ages}
# stds = {val: np.std(data.loc[(data.age >= val) & (data.age < val+10), 'charges']) for val in ages}
# for val in ages:
#     data.loc[(data.age >= val) & (data.age < val+10), 'charges'] = (data.loc[(data.age >= val) & (data.age < val+10), 'charges'] - means[val]) / stds[val]
dum = pd.get_dummies(data['region'])
data = data.drop('region', 1)
data = pd.concat((data, dum), axis=1)
print(data.head(10))

# corr = data.corr()
# print(corr)
# sns.pairplot(data, size=2.5)
# plt.show()
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 従属変数と独立変数を分離
data_X = data.drop('charges', 1).drop('smoker', 1)
data_y = data.charges

# 学習データと検証データに分ける(クロスバリデーション)
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y)
model = LinearRegression()
model.fit(X_train, y_train)

# 学習精度
pred_train = model.predict(X_train)
train_error = np.sqrt(np.mean((y_train-pred_train) ** 2))
# 予測精度
pred_test = model.predict(X_test)
test_error = np.sqrt(np.mean((y_test-pred_test) ** 2))
# 結果の表示
print('smokerを外した場合')
print('学習誤差：%f'%train_error)
print('予測誤差：%f'%test_error)

# 従属変数と独立変数を分離
data_X = data.drop('charges', 1)
data_y = data.charges

# 学習データと検証データに分ける(クロスバリデーション)
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
model = LinearRegression()
model.fit(X_train, y_train)

# 学習誤差
pred_train = model.predict(X_train)
train_error = np.sqrt(np.mean((y_train-pred_train) ** 2))
# 予測誤差
pred_test = model.predict(X_test)
test_error = np.sqrt(np.mean((y_test-pred_test) ** 2))
# 結果の表示
print('smokerを外さなかった場合')
print('学習誤差：%f'%train_error)
print('予測誤差：%f'%test_error)
# 可視化
train = plt.scatter(pred_train, (pred_train - y_train), c= 'b', alpha=0.5) 
test = plt.scatter(pred_test, (pred_test - y_test), c ='r', alpha=0.5)
plt.hlines(y = 0, xmin = -1.0, xmax = 2)
# 凡例
plt.legend((train, test), ('Training', 'Test'), loc = 'lower left') # 凡例
# タイトル(残差プロット)
plt.title('Residual Plots')
plt.show()

plt.hist((pred_train - y_train))
plt.hist((pred_test - y_test))
print('誤差の平均値：%f'%np.mean((pred_test - y_test)))
print('誤差の標準誤差：%f'%np.std((pred_test - y_test)))
from sklearn.decomposition import PCA

decomposer = PCA(n_components=2)
#decomposer.fit(X_train.drop('age', 1))
decomposer.fit(data)

#平均ベクトル(D次元ベクトル)
M = decomposer.mean_
#print(M)
print("主成分の分散説明率")
print(decomposer.explained_variance_ratio_)

#主成分ベクトル（主成分数xDの行列）
V = decomposer.components_
print(V)

#固有値（各主成分におけるデータの分散）
E = decomposer.explained_variance_
print(E)

# 分析結果を元にデータセットを主成分に変換する
#transformed = decomposer.fit_transform(X_train.drop('age', 1))
transformed = decomposer.fit_transform(data)
print(transformed)

# 主成分をプロットする
plt.figure(figsize=(10,10))
plt.subplot(2, 1, 1)
labels=[idx for idx in range(4)]
plt.scatter(transformed[:,0], transformed[:,1], c=pd.cut(data.charges, 4, labels=labels))
plt.title('principal component')
plt.xlabel('pc1')
plt.ylabel('pc2')
from sklearn.decomposition import FastICA

#独立成分の数＝2
decomposer = FastICA(n_components = 2)

#データの平均を計算
M = np.mean(data.T, axis = 1)[:,np.newaxis]
print(M)
print(data.head(2))

#各データから平均を引く
data2 = data.copy()
for idx in range(len(M)):
    data2.iloc[:, idx] = data.iloc[:, idx] - M[idx]
#print(data2.head(2))

#平均0としたデータに対して、独立成分分析を実施
decomposer.fit(data2)

#独立成分ベクトルを取得(D次元 x 独立成分数)
transformd = decomposer.transform(data2)
print(transformed)

# 独立成分をプロットする
plt.figure(figsize=(10,10))
plt.subplot(2, 1, 1)
labels=[idx for idx in range(4)]
plt.scatter(transformed[:,0], transformed[:,1], c=pd.cut(data2.charges, 4, labels=labels))
plt.title('independence component')
plt.xlabel('ic1')
plt.ylabel('ic2')