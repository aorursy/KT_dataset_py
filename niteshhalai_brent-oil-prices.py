# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/brent-oil-prices/BrentOilPrices.csv')

data
data['Date'] = pd.to_datetime(data['Date'])
data['last_price'] = 9.12

data
data.set_index('Date', inplace=True)
fig, ax=plt.subplots(figsize=(15,7))

plt.title('Brent Oil Prices', fontsize='xx-large')

plt.style.use('ggplot')

plt.xlabel('Date')

plt.ylabel('Price (USD)')

ax.plot(data.index, data['Price'], label='price')

ax.plot(data.index, data['last_price'], label='last_price')

ax.plot()

plt.show()