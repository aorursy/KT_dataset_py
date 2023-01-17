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
#read CSV file into a dataframe



df = pd.read_csv('/kaggle/input/big-five-stocks/big_five_stocks.csv')
#display last 100 rows



df.tail(100)
# plot NASDAQ Composite index (opening, closing, highest, and lowest) in the last 90 days until 8/23/2019



df[df['name'] == '^IXIC'][['open', 'close', 'high', 'low' ]].tail(90).plot()
# plot all historical stock prices (opening, closing, highest, and lowest) for Amazon

# use NASDAQ stock ticker for masking, i.e. df['name'] == 'AAPL' for apple



df[df['name'] == 'AMZN'][['open', 'close', 'high', 'low' ]].plot()
# plot stock prices (opening, closing, highest, and lowest) for Alphabet, co (google) in the last 90 days till 8/23/2019



df[df['name'] == 'GOOGL'][['open', 'close', 'high', 'low' ]].tail(90).plot()