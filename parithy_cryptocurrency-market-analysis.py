# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/crypto-markets.csv")
df.head()
dfBTC=df[df['symbol']== 'BTC']

df2=dfBTC.drop(['slug','name','ranknow','high','low','close_ratio','spread'],axis=1)
df2['date'] = pd.to_datetime(df2.date)
df2.set_index('date', inplace=True)
print(df2.tail())
print(df2.dtypes)
df2.isnull().any()
sns.pairplot(df2[['close','volume','market']])

sns.heatmap(df2.loc[:,['close','volume','market']].corr(),annot=True)
df2.head().hist()