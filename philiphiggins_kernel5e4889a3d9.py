!pip install tslearn
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tslearn as ts
import numpy as np
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
items=pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
transactions=pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
categories=pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
shops=pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
cumulative=list(['date',])
for i in range(shops.shop_id.size):
    cumulative=cumulative+[str(i)]
dfCumul=pd.DataFrame(columns=cumulative)
dfCumul

date=0
shop_id=-1
prices=[0]*(shops.shop_id.size)
for i in transactions.sort_values(by=['date','shop_id']).itertuples():
    if i.date!=date:
        dfCumul=dfCumul.append(pd.Series([i.date]+prices, index=dfCumul.columns), ignore_index=True)
        date=i.date
        prices=[0]*(shops.shop_id.size)
    prices[i.shop_id]=prices[i.shop_id]+i.item_price
dfCumul #we now have the total cost for each shop each day
