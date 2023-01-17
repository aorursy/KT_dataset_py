import os



# 慣例的に次のようにインポートして利用することが多い

import pandas as pd
INPUT_DIR = '../input/ailab-ml-training-2'



TRAIN_DATA_PATH = os.path.join(INPUT_DIR, 'train.csv')

TEST_DATA_PATH = os.path.join(INPUT_DIR, 'test.csv')

SUBMIT_DATA_PATH = os.path.join(INPUT_DIR, 'sample_submission.csv')
# read_csv()関数で .csv ファイルを DataFrame として読み込む

train_df = pd.read_csv(TRAIN_DATA_PATH)

test_df = pd.read_csv(TEST_DATA_PATH)

sample_submission_df = pd.read_csv(SUBMIT_DATA_PATH)
print(f'train data shape: {train_df.shape}')

train_df.head()
# テストデータも確認してみましょう

# 目的変数・targetである price が欠けていることがわかります．

# また，　`train_id`の代わりに`test_id`がふられています．

print(f'test data shape: {test_df.shape}')

test_df.head()
# 最後に提出フォーマットも確認します

# 各 `test_id` に対して予測した価格を記載する方式であることがわかります

print(f'sample submission shape: {sample_submission_df.shape}')

sample_submission_df.head()
print(f'shapeの確認: {train_df.shape}')

print('----------------------------------------------------------------------')



print(f'index(行名)の確認: {train_df.index}')

print('----------------------------------------------------------------------')



print(f'column(列名)の確認: {train_df.columns}')

print('----------------------------------------------------------------------')



print(f'データの型を確認: {train_df.dtypes}')
# 任意の列を抽出して表示

train_df[['name', 'price', 'item_description']].head()
# 任意の行を抽出

train_df[5:10]
# 条件を指定して抽出

# 他にも `query`を利用することでSQLライクな操作も可能となっている

train_df[train_df['item_condition_id'] == 3].head()
# 1の例

train_df.loc[train_df['item_condition_id'] == 3, 'price']
# 2の例

# 次の例では， `train_df`の全ての行に対して (:を使用．スライス表記が可能)，`train_id`から`category_name`までを抽出している．

# 注: 一般的なスライス表記とは含有される領域が異なることに注意 (list[3:5]とかくと，通常ならindex=5は除外される)

train_df.loc[:, 'train_id':'category_name']
# 3の例

# iloc系では通常のスライス表記よろしく最後は含まれない， loc系だけが少し特殊になっている

# もちろん行の指定ではなく，列を指定することもできるが，可読性が落ちるため非推奨．

train_df.iloc[4:8]
# unique()を使うことで，重複なしでどのような値が入っているのか調べることができる

# `item_condition_id`は1, 2, 3, 4, 5 の5種類の値があることがわかる

train_df['item_condition_id'].unique()
# 統計量をまとめて表示してくれる

# 計算できない列，例えば `item_description`などは表示されていない

train_df.describe()
# 欠損地の個数や，データの型などがわかる

train_df.info()
# 自動的にふられていた index の代わりに， `train_id` をindexとして使用する

# 

# train_df.set_index('train_id', inplace=True) とすると，中身を実際に書き換える

# train_df = train_df.set_index('train_id') と同じ操作↑



train_df.set_index('train_id')  # 実際にデータ自体は変えず，適応結果を確認することができる
# カラム名の変更

# こちらも inplace=True とすると実際に書き換わる

train_df.rename(columns={

    'category_name': 'category',

    'item_description': 'description'

})
# カラムの値に応じて，データを並び替える

# ここでは価格順に並び替えてみます．

# byに渡す引数をリストにすると，複数の列を指定してソートできる(例えば，item_condition_idごとにprice順に並べるなど)

# inplace=Trueとすると．．．



train_df.sort_values(by='price', ascending=False).head()
# 欠損データ数の確認は df.info()からもわかりますが，こんな便利な関数もあります

# df.inull() : NaNの箇所をTrue(1), それ以外の値が入っている箇所をFalse(0)とする



train_df.isnull().sum()
# 欠損値をどう扱えば良いのか？

# 特にモデルとして Neural Networks を用いる場合，欠損値をモデル側で処理できないため，ユーザが的確に処理を行う必要があります

# いろいろな方法が考えられますが，単純なものだと，平均値で埋める，0で埋めるなどといった手法があります

# inplaceすると...



# 動作を調べるために`brand_name`の中の`NaN`を`None`に置き換えてみます



train_df.fillna(value={'brand_name': 'None'}).head()
# NaNを含むデータは捨ててしまう(学習に利用しない)という手法も考えられます

# inplace...



train_df.dropna(subset=['brand_name'], axis=0)
# columnごと削除したい際は dropをaxis=1方向で利用します



train_df.drop(['category_name'], axis=1).head()
type(train_df['price'])
tmp_series = train_df['price']
tmp_series.head()
# seriesの各要素に対してある関数を適用したい際に重宝するのが apply()関数です

# 例えば，log1p関数を全体に適用したいと思ったら

import numpy as np

tmp_series.apply(np.log1p).head()
# もちろんlambda式も使える

tmp_series.apply(lambda x: x+10).head()
# brand_nameを抜き出したSeriesを作り直します

tmp_series = train_df['brand_name']
# Nike, PINK, など，各ブランドごとの商品数を獲得することができました

tmp_series.value_counts()