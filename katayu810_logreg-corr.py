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
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

train.head()
sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

sample_submission.head()
import seaborn as sns

import matplotlib.pyplot as plt

corrmat = train.corr()

f, ax = plt.subplots(figsize =(10, 10))

sns.heatmap(corrmat, vmax = .8, square = True)
k = 10

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

#相関係数高い順top10を求めた



sns.set(font_scale = 1.25)

cm = sns.heatmap(cm, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

#annot=セルに値を出力

#square=X,yで正方形

#fmt = 文字のフォーマットを指定

#annot_kws = 文字のサイズ



plt.show()

cols
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars',

       'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], size=2.5)

plt.show()
total = train.isnull().sum().sort_values(ascending=False)

percent= (train.isnull().sum()/train.isnull().count()).sort_values(ascending = False)

#df_train.isnull()は全て判定



missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])

missing_data
train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)

train
train = train.drop(train.loc[train['Electrical'].isnull()].index)

train.isnull().sum().max()
train
from sklearn.model_selection import train_test_split



# 目的変数

y = train['SalePrice']



#説明変数(行列は大文字にする)

X = train[['OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea','FullBath']]



# 学習用、検証用データに分割

(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size = 0.2, random_state = 5)
from sklearn import linear_model



model = linear_model.LinearRegression()

model.fit(X,y)
from sklearn.metrics import r2_score

y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)



print("R2_y_train{}".format(r2_score(y_train,y_train_pred)))

print("R2_y_test", r2_score(y_test,y_test_pred))

#テストの方が良いのでか学習を起こしていない
plt.scatter(y_train, y_train_pred)

plt.scatter(y_test, y_test_pred, c = 'red')
test.isnull().sum().sort_values(ascending = False)

test["TotalBsmtSF"].isnull().sum()
index = test[test['TotalBsmtSF'].isnull()].index

index
test['TotalBsmtSF'][index[0]] = test['TotalBsmtSF'].mean()
X_sub = test[['OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea','FullBath']]



'''予測値の算出'''

y_sub = model.predict(X_sub)

y_sub
'''予測結果の提出用ファイルを作成する'''

#IDと予測結果を結合する

df_sub_pred = pd.DataFrame(y_sub).rename(columns={0:'SalePrice'})

print(df_sub_pred)

df_sub_pred = pd.concat([test['Id'], df_sub_pred['SalePrice']], axis=1)

df_sub_pred

# CSVファイルの作成 

df_sub_pred.to_csv("Submit.csv", index=False)