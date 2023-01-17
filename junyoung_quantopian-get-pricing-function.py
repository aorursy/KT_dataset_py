# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.

code_df = pd.read_csv('../input/code.csv')
code_df.code = code_df.code.astype(str)
code_df.code = code_df.code.str.zfill(6)

kospi_df = pd.read_csv('../input/kospi.csv')
kospi_df.name = kospi_df.name.astype(str)
kospi_df.name = kospi_df.name.str.zfill(6)
kospi_df.info()
def get_stock_by_code(code: str, field: str, start_date: str, end_date: str):
    df = kospi_df[kospi_df.name == code]
    df = df[df.date.between(start_date, end_date, inclusive=True)]
    df.index = df.date
    return df[field]

def get_stock_by_name(name: str, field: str, start_date: str, end_date: str):
    if not code_df.corp.isin([name]).any():
        raise ValueError('Cannot find {} in codes'.format(name))
    code = code_df[code_df.corp == name].code.tolist()[0]
    df = get_stock_by_code(code, field, start_date, end_date)
    return df
df = get_stock_by_code('005930', 'close', '2018-09-01', '2018-10-01')
df.head()
df = get_stock_by_name('삼성전자', 'close', '2018-09-01', '2018-10-01')
df.head()
df = get_stock_by_name('심술전자', 'close', '2018-09-01', '2018-10-01')
df.head()
