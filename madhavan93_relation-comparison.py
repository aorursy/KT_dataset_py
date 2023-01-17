# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np #p;p0 linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

# print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

files_data = []

for file in os.listdir("../input"):

    file_data = pd.read_csv('../input/' + file)

    files_data.append(file_data)



stores_data = pd.concat(files_data)



stores_data['Discount'] = stores_data['MRP'] - stores_data['Sales Price']

stores_data['Discount Rate'] = np.where( stores_data['MRP'] != 0, 

                    (stores_data['Discount'] * 100) / stores_data['MRP'], 0)

    

stores_data.loc[(stores_data['MRP'] == 0) & (stores_data['Sales Price'] != 0), 'MRP Anamoly'] = True

stores_data.loc[~((stores_data['MRP'] == 0) & (stores_data['Sales Price'] != 0)), 'MRP Anamoly'] = False



stores_data['Sales Anamoly'] =  stores_data['Sales Price'] > stores_data['MRP']



stores_data['Sale Date'] = stores_data['Sale Date'].astype('datetime64[ns]')

stores_data['Month'] = pd.DatetimeIndex(stores_data['Sale Date']).month

stores_data['Year'] = pd.DatetimeIndex(stores_data['Sale Date']).year



margin = 0.5

stores_data['Margin'] = stores_data['Sales Price'] * margin

stores_data['Margin Quantity'] = stores_data['Margin'] * stores_data['Sales Qty']



stores_data.head()
def wavg(group, avg_name, weight_name):

    d = group[avg_name]

    w = group[weight_name]

    try:

        return (d * w).sum() / w.sum()

    except ZeroDivisionError:

        return 0
grouped_by_brand = stores_data.groupby('Brand Code')

aggregator = {'Category': pd.Series.nunique, 'Store Code': pd.Series.nunique, 'Month': pd.Series.nunique, 'Sale/Return': Counter}

grouped_by_brand.agg(aggregator).sort_values('Category', ascending=False)