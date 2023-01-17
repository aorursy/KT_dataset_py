import numpy as np

import pandas as pd
# データを読み込む

df = pd.read_csv('/kaggle/input/machine-learning-homework/train.csv', index_col=0)
# 内容確認

df.Municipality
# カンマ区切りで連なる個数は最大２個であることを確認

df.Municipality.apply(lambda x: len(x.split(','))).max()
# 1ブロック目の末尾を取り出す

Municipality_0 = df.Municipality.apply(lambda x: x.split(',')[0].split(' ')[-1])

Municipality_0
# 2ブロック目の末尾を（ブロック数が２の場合に）取り出す。

Municipality_1 = df.Municipality.apply(lambda x: x.split(',')[1].split(' ')[-1] if len(x.split(','))==2 else '')

Municipality_1
# 両者を結合する。

df['Municipality_type'] = Municipality_0 + Municipality_1

df['Municipality_type']