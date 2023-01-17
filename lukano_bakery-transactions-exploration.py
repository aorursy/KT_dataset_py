# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline
from datetime import datetime
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# let's get an overview over the structure of the data
data = pd.read_csv("../input/BreadBasket_DMS.csv")
data.head(10)
data['Date'].describe()
data['Transaction'].describe()
# what items are sold in the bakery?
for item in set(data['Item']):
    print(item)
print('There are ' + str(len(set(data['Item']))) + " items that can be purchased in the bakery.")
Item = []
Purchases = []
for item in set(data['Item']):
    Item.append(item)
    indices = [i for i, x in enumerate(data['Item']) if x == item]
    Purchases.append(len(indices))
df_purchase = pd.DataFrame(data = [Item, Purchases])
df_purchase = df_purchase.transpose()
df_purchase.columns = ['Item', 'Purchases']
df_purchase.sort_values(by=['Purchases'], inplace=True)
df_purchase.head(10)

plt.figure(figsize=(15,15))
sns.barplot(x=df_purchase['Purchases'], y=df_purchase['Item'])
# when are these items sold?
items = np.transpose(np.array(list(set(data['Item']))))
purchase_time = pd.DataFrame(np.zeros((len(items), 24)))


zipped = zip(range(95), items)
item_dict = {}
for ID, item in zipped:
    item_dict[ID] = item
  
purchase_time.rename(item_dict, axis='index', inplace=True)
purchase_time.head()
for item in items:
    indices = [i for i, x in enumerate(data['Item']) if x == item]
    for index in indices:
        time = data['Time'].iloc[index][0:2]
        purchase_time.loc[[item], [int(time)]] = purchase_time.loc[[item], [int(time)]] + 1
        
purchase_time.head(10)
plt.figure(figsize=(25,25))
for i in range(1, len(items)):
    plt.subplot(19, 5, i)
    sns.lineplot(x=range(0,24), y=purchase_time.loc[items[i-1]])
# coming next: correlations between item to be sold and date/time.