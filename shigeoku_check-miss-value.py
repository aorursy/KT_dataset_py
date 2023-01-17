import sys
import numpy as np
import pandas as pd

# args
# df：欠損値を確認する DataFrame
# returns
# df: 欠損値を確認した結果 DataFrame

def check_miss_value(df):
    null_value = df.isnull().sum()
    miss_percent = 100 * df.isnull().sum() / len(df)
    miss_table = pd.concat([null_value, miss_percent], axis=1)
    miss_table_columns = miss_table.rename(
    columns = {0: '欠損数', 1: '%'})
    miss_table.coulmns = ['欠損数', '%']
    return miss_table_columns