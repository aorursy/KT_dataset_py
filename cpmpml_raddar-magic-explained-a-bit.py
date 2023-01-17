# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
historical_transactions = pd.read_csv('../input/historical_transactions.csv', usecols=['purchase_amount'])

new_transactions = pd.read_csv('../input/new_merchant_transactions.csv', usecols=['purchase_amount'])

data = pd.concat((historical_transactions, new_transactions))
data.purchase_amount.mean()
data['new_amount'] = (data.purchase_amount - data.purchase_amount.min())
s = data.groupby('new_amount').new_amount.first().to_frame().reset_index(drop=True)

s.head(10)
s['delta'] = s.new_amount.diff(1)

s.head(10)
s[s.delta > 2e-5].head()
s = s[1:52623]
s.head()
s.tail()
s.delta.mean()
data['new_amount'] = data.new_amount / (100 * s.delta.mean())
data.new_amount.value_counts().head(10)
data['two_decimal_amount'] = np.round(data.new_amount, 2)
np.abs(data.two_decimal_amount - data.new_amount).mean()
(data.two_decimal_amount == np.round(data.two_decimal_amount)).mean()
tmp = data[-new_transactions.shape[0]:]

(tmp.two_decimal_amount == np.round(tmp.two_decimal_amount)).sum()