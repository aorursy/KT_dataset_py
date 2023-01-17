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
###Step 1: 检视源数据集
train_df = pd.read_csv('../input/train.csv', index_col=0)
test_df = pd.read_csv('../input/test.csv', index_col=0)
train_df.head() ##默认看前5行
#####Step 2: 合并数据
###将数据归一化成数字化表示，要将所有的数据都进行处理，包括train、test
##处理y，label本身并不平滑。为了我们分类器的学习更加准确，我们会首先把label给“平滑化”（正态化）
#log1p 为 log(x+1)
#这里先看看长什么样子，没有做处理
#%matplotlib inline
prices = pd.DataFrame({"price":train_df["SalePrice"], "log(price + 1)":np.log1p(train_df["SalePrice"])})
prices.hist()
#将训练集的售价取出，方便接下来合并两个集合
y_train = np.log1p(train_df.pop('SalePrice'))
#合并两个数据集
all_df = pd.concat((train_df, test_df), axis=0)
###看看合并后的结果
all_df.shape
####Step 3: 变量转化
##类似『特征工程』。就是把不方便处理或者不unify的数据给统一了
###看数据的描述，明确数据的属性  有些数字可能只表示类别，我们需要区分一下
#MSSubClass 的值其实应该是一个category
all_df['MSSubClass'].dtypes
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)
all_df['MSSubClass'].value_counts()
###把category的变量转变成numerical表达形式

#当我们用numerical来表达categorical的时候，要注意，数字本身有大小的含义，所以乱用数字会给之后的模型学习带来麻烦。
#于是我们可以用One-Hot的方法来表达category。
#pandas自带的get_dummies方法，可以帮你一键做到One-Hot
pd.get_dummies(all_df['MSSubClass'], prefix='MSSubClass').head()
###把所有的是分类的数据都变成数值型形式，pandas的get_dummies函数会自动识别是类别的列，并转换成one-hot形式
all_dummy_df = pd.get_dummies(all_df)
all_dummy_df.head()
###处理好numerical变量（数字缺失）

##查看有哪些有缺失
all_dummy_df.isnull().sum().sort_values(ascending=False).head(10)
         ##看看怎么填充缺失值（平均值或者其他）
#求平均值
mean_cols = all_dummy_df.mean()
mean_cols.head(10)
#用平均值填充
all_dummy_df = all_dummy_df.fillna(mean_cols)
##看看还有没有缺失
all_dummy_df.isnull().sum().sum()
###标准化numerical数据

   ###先来看看 哪些是numerical的
numeric_cols = all_df.columns[all_df.dtypes != 'object']
numeric_cols

##计算标准分布：(X-X')/s
#让我们的数据点更平滑，更便于计算。
numeric_col_means = all_dummy_df.loc[:, numeric_cols].mean()
numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()
all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std

####Step 4: 建立模型
##11把数据集分回 训练/测试集
dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]
dummy_train_df.shape, dummy_test_df.shape
###Ridge
from sklearn.linear_model import Ridge   ##Ridge Regression模型会把所有的变量都放进分类器，给出结果
from sklearn.model_selection import cross_val_score
###把数据转成了numpy形式，方便处理
X_train = dummy_train_df.values
X_test = dummy_test_df.values
alphas = np.logspace(-3, 2, 50)
test_scores = []
for alpha in alphas:
    clf = Ridge(alpha)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
#存下所有的CV值，看看哪个alpha值更好（也就是『调参数』）
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(alphas, test_scores)
plt.title("Alpha vs CV Error");
###可见，大概alpha=10~20的时候，可以把score达到0.135左右
###Random Forest
from sklearn.ensemble import RandomForestRegressor

max_features = [.1, .3, .5, .7, .9, .99]
test_scores = []
for max_feat in max_features:
    clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
plt.plot(max_features, test_scores)
plt.title("Max Features vs CV Error");
##在0.3的时候最优
##Step 5: Ensemble  我们用一个Stacking的思维来汲取两种或者多种模型的优点
ridge = Ridge(alpha=15)
rf = RandomForestRegressor(n_estimators=500, max_features=.3)

ridge.fit(X_train, y_train)
rf.fit(X_train, y_train)
##因为最前面我们给label做了个log(1+x), 于是这里我们需要把predit的值给exp回去，并且减掉那个"1"
##所以就是我们的expm1()函数。
y_ridge = np.expm1(ridge.predict(X_test))
y_rf = np.expm1(rf.predict(X_test))

###一个正经的Ensemble是把这群model的预测结果作为新的input，再做一次预测。这里我们简单的方法，就是直接『平均化』
y_final = (y_ridge + y_rf) / 2
##Step 6: 提交结果
submission_df = pd.DataFrame(data= {'Id' : test_df.index, 'SalePrice': y_final})
submission_df.head(10)
submission_df.to_csv("logistic_regression_predictions.csv", index=False)