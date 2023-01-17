# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dfb = pd.read_csv('/kaggle/input/binance-real-time-trades-btcusdt-ethusdt/BTCUSDT.csv', dtype={'time': np.int64, 'id': np.uint64, 'price': np.float32, 'qty': np.float32, 'isBuyerMaker':np.int32})

print(dfb.size)

dfe = pd.read_csv('/kaggle/input/binance-real-time-trades-btcusdt-ethusdt/ETHUSDT.csv', dtype={'time': np.int64, 'id': np.uint64, 'price': np.float32, 'qty': np.float32, 'isBuyerMaker':np.int32})

print(dfe.size)
b0 = dfb.iloc[:, 0].to_numpy()

b2 = dfb.iloc[:, 2].to_numpy()



e0 = dfe.iloc[:, 0].to_numpy()

e2 = dfe.iloc[:, 2].to_numpy()



print(b2)

print(e2)
import matplotlib.pyplot as plt

%matplotlib inline



plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')

plt.plot(b0, b2, "-")

plt.xlabel("Time")

plt.ylabel("Price")

plt.grid(False)
plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')

plt.plot(e0, e2, "-")

plt.xlabel("Time")

plt.ylabel("Price")

plt.grid(False)