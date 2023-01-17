import numpy as np

import pandas as pd

from pandas import DataFrame
# DataFrameを作っておく。例えばここでは3<=x<5の範囲を削除したいとする。

df = DataFrame(np.arange(10), columns=['x'])

df
# 削除したい条件を指定する。ブーリアンのSeriesになっているのがわかる。

condition = (df['x']>=3)&(df['x']<5)

condition
# 裏返すと残したい条件になる

~condition
# 出来たブーリアンのSeriesをフィルターにして出来上がり

df_select = df[~condition]

df_select
# queryを用いてもうちょっと直感的に書くこともできるが、その辺りは好みで。

df_select = df.query('x<3 or x>=5')

df_select