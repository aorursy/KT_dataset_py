%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.api as sm
import pylab as pl

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df.describe(include='all')
df.dropna()
train_cols = df.columns[3:]
train_cols
df['QuoteConversion_Flag'].hist()
pl.show()
df['QuoteConversion_Flag'] = df['QuoteConversion_Flag'].astype(int)
#logit = logit = sm.Logit(df['admit'], df[train_cols])
#for column in df.columns.values.tolist():
#    print(column + ':' + str(type(df[column])))
logit = sm.Logit(df['QuoteConversion_Flag'], df[['Field7','Field8','Field9']])
results = logit.fit()
results.summary()