#参考　
# https://qiita.com/teru855/items/8346a94abde86a842a1b


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# データを入力する　macでバックスラッシュはoption+¥
df= pd.read_csv("../input/train.tsv", delimiter="\t")
df.head()

# Any results you write to the current directory are saved as output.
#  ブランドの種類
len(df.brand_name.unique())
# 価格の種類
len(df.price.unique())
# コンディションの種類
df.item_condition_id.unique()

# カテゴリごとの、最高価格、最低価格を調べる
#カテゴリは大分類、中分類、小分類になっているはず
#アプローチは、分類を数値化して一番似ている文字の値付けを真似る感じだろうな
#カテゴリごとのテーブルをもてばいいのでは

# アイテムの種類
len(df.train_id.unique())

# 100万件のデータがあるってことか、すごいな
# 最大2000ドルか、、、20万円ってことだな
df.price.max()
# 最低価格は、無料ってこと
df.price.min()

s = df['price'].value_counts()
print(s)

# 価格分布をみて、もっとも当てはまりそうな価格とのマッチングをみればいいんじゃないかな、だいたいの価格がわかればいいんでしょ

# 値が0の要素数
s = df['price'].value_counts()
#len(s)

s.head(60)


df.price.describe()
df["item_description"]
# 価格ごとに並び替える
# 出現順に１０段階にわける
#　そのレベルの特徴量を出す　説明文のテキスト、内容、とか
#　そのレベルの特徴量って何かわかるものなのかな
#　単語の羅列を入れてみて、何かわかるのか？
#　組み合わせなのか、
#　使われている単語の、ヒストグラムをつくる？のか
#　なんの違いも生まれないんじゃないかな
#　道の単語によって、shipping のアイテムの重要度が変わってくる感じだな
# そんなアルゴリズムをどう書けばいいのだ？
# 巨大な単語超から、キーワードごとの値を出すところなんじゃないか

a= pd.read_csv("../input/sample_submission.csv")
a.head()
a.tail()

#df.loc[df.name == 'socks', 'price'] = 500
a.loc[a.test_id == 1, 'price'] = 10
a.head()
#df.loc[df.name == 'socks', 'price'] = 500
a.loc[a.test_id != -1, 'price'] = 10
a.head()
#a.to_csv("mercari_1.csv")
a.to_csv("mercari_2.csv", index=False)