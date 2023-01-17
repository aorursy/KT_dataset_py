# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
tesla = pd.read_csv("../input/tesla-stock-price/Tesla.csv - Tesla.csv.csv")
tesla['MA10'] = tesla['Close'].rolling(10).mean()
tesla['MA50'] = tesla['Close'].rolling(50).mean()
tesla = tesla.dropna()
tesla.head()


plt.figure(figsize=(10, 8))
tesla['MA10'].loc['1':'100'].plot(label='MA10')
tesla['Close'].loc['1':'100'].plot(label='Close')
tesla['MA50'].loc['1':'100'].plot(label='MA50')
plt.legend()
plt.show()
tesla['Shares'] = [1 if tesla.loc[ei, 'MA10']>tesla.loc[ei, 'MA50'] else 0 for ei in tesla.index]
tesla['Close1'] = tesla['Close'].shift(-1)
tesla['Profit'] = [tesla.loc[ei, 'Close1'] - tesla.loc[ei, 'Close'] if tesla.loc[ei, 'Shares']==1 else 0 for ei in tesla.index]
tesla['Profit'].plot()
plt.axhline(y=0, color='red')
tesla.head()
tesla['wealth'] = tesla['Profit'].cumsum()
tesla.tail()
tesla['wealth'].plot()
plt.title('Total money you win is {}'.format(tesla.loc[tesla.index[-2], 'wealth']))
