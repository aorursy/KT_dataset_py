# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print('Printing Only Few Files..')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames[:5]:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# registerting Helper Module Path

import sys

sys.path.append('/kaggle/input/stock-market-india')
import StockMarketHelper as helper
helper.read_master()
# searching stock

helper.filter_symbol(search='NIFTY', return_all=True)
helper.filter_symbol(search='NIFTY', return_all=False)
helper.read_from_key('NIFTY_50__EQ__INDICES__NSE__MINUTE')
helper.read_data(search='AI')
helper.read_data(query='BHARTIARTL')