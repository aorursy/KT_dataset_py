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
import pandas as pd
hist = pd.read_csv('../input/historical_transactions.csv')
hist.info(memory_usage='deep')
hist.head()
hist.authorized_flag.value_counts()
hist.authorized_flag = hist.authorized_flag.astype('category')
hist.info(memory_usage='deep')
hist.category_1.value_counts()
hist.category_1 = hist.category_1.astype('category')
hist.category_2.value_counts()
hist.category_2 = hist.category_2.astype('category')
hist.category_3.value_counts()
hist.category_3 = hist.category_3.astype('category')
hist.info(memory_usage= 'deep')
hist.city_id.max(),hist.city_id.min(),sum(hist.city_id.isnull())
hist.city_id = hist.city_id.astype('int16')
hist.city_id.max(),hist.city_id.min(),sum(hist.city_id.isnull())
hist.installments.max(),hist.installments.min(),sum(hist.installments.isnull())
hist.installments = hist.installments.astype('int16')
hist.installments.max(),hist.installments.min(),sum(hist.installments.isnull())
hist.month_lag.max(),hist.month_lag.min(),sum(hist.month_lag.isnull())
hist.month_lag = hist.month_lag.astype('int16')
hist.month_lag.max(),hist.month_lag.min(),sum(hist.month_lag.isnull())
hist.state_id.max(),hist.state_id.min(),sum(hist.state_id.isnull())
hist.state_id = hist.state_id.astype('int16')
hist.state_id.max(),hist.state_id.min(),sum(hist.state_id.isnull())
hist.subsector_id.max(),hist.subsector_id.min(),sum(hist.subsector_id.isnull())
hist.subsector_id = hist.subsector_id.astype('int8')
hist.subsector_id.max(),hist.subsector_id.min(),sum(hist.subsector_id.isnull())
hist.info(memory_usage='deep')
unique_merchants = pd.DataFrame(hist.merchant_id.value_counts())
unique_merchants.reset_index(inplace = True)
unique_merchants.reset_index(inplace = True)
unique_merchants.columns = ['merchant_id_code','merchant_id','count']
unique_merchants.drop(columns = ['count'],inplace = True)
unique_merchants.head()
sum(hist.merchant_id.isnull())
hist = hist.merge(unique_merchants, on = 'merchant_id',how = 'left')
hist.drop(columns = ['merchant_id'],inplace = True)
hist.merchant_id_code.fillna(value = -1,inplace = True)
hist.merchant_id_code = hist.merchant_id_code.astype('int32')
unique_cards = pd.DataFrame(hist.card_id.value_counts())
unique_cards.reset_index(inplace = True)
unique_cards.reset_index(inplace = True)
unique_cards.columns = ['card_id_code','card_id','count']
unique_cards.drop(columns = ['count'],inplace = True)
unique_cards.head()
hist = hist.merge(unique_cards, on = 'card_id',how = 'left')
hist.drop(columns = ['card_id'],inplace = True)
hist.card_id_code.fillna(value=-1,inplace = True)
hist.card_id_code = hist.card_id_code.astype('int32')
hist.info(memory_usage='deep')
unique_merchants.info(memory_usage='deep')
unique_cards.info(memory_usage='deep')
