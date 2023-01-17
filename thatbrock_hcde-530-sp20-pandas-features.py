# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

s = pd.Series([1,2,3, np.nan, 5, 6])
s
dates = pd.date_range('20200101', periods = 6)
dates
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
df
df.tail(1)
df.index
df.describe()
df.T
df
df.iloc[[1,1]]
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2020', periods=1000))
cum = ts.cumsum()
cum.plot()