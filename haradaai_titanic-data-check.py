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
import pandas as pd

df = pd.read_csv("/kaggle/input/titanic/train.csv")



# 項目の確認

df.head()
# 統計量の確認

df.describe()
# ヒストグラム

df.hist(figsize=(12,12))
import seaborn as sns

# 性別と生存率

sns.countplot('Sex', hue = 'Survived', data=df)

# 男の人の方が生き残っている
# 乗車料金と生存率

sns.countplot('Fare', hue = 'Survived', data=df)

# 細かすぎて見辛い....
# 年齢と生存率

sns.countplot('Age', hue = 'Survived', data=df)

# そこまで顕著な違いは見えない...?
# TODO

# 学習できるように、文字データを数字にする！ （male→１ female→０ みたいな。）

# 細かい数字を四捨五入する12.3とかを整数にする
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")



full_data = [train,test]



train.head(2)
for data in full_data:

    data['Sex'] = data['Sex'].map({'female':0, 'male':1}).astype(int)

train.head(2)


# # 年齢が細かすぎるので、わける



for dataset in full_data:

    dataset.loc[dataset['Age'] <= 20, 'Age' ] = 0

    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 30), 'Age' ] = 1

    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 40), 'Age' ] = 2

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 50), 'Age' ] = 3

    dataset.loc[(dataset['Age'] > 50) & (dataset['Age'] <= 60), 'Age' ] = 4

    dataset.loc[(dataset['Age'] > 60) , 'Age' ] = 5

    dataset.loc[dataset['Age'].isnull(),'Age'] = 6;

            

train.head(6)
# 年齢と生存率

sns.countplot('Age', hue = 'Survived', data=train)

# 高齢だからとか、若いからとかと生死にそこまで大きなさはなさそう....
# 生死に関係なさそうなデータを消す

x_train = train.drop(['Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)

x_train.head(10)
# 正解ラベルを作っておく

y_train = df['Survived'].values
from sklearn.linear_model import LogisticRegression

import sklearn.preprocessing as sp



# 今回は２値分類なので、ロジスティック回帰を利用

model = LogisticRegression()

model.fit(x_train,y_train)

model.score(x_train,y_train)
# テストデータも欠損データの補完をする

test.head()
# 欠損値に０を入れる 

test = test.fillna(0)

test.head()