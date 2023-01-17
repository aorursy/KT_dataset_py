# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

dtrain = pd.read_csv('../input/train.csv')
print(dtrain.columns)
import matplotlib.pyplot as plt
import seaborn as sns
df = dtrain[dtrain.item_seq_number < 20].copy()
plt.figure(figsize = (12, 8))
sns.barplot('item_seq_number','deal_probability', data=df)
plt.ylabel('mean_deal_probability')
plt.xlabel('item_seq_number')
plt.title('item_seq_number vs mean_deal_probability')
plt.show()
df = dtrain[dtrain.item_seq_number < 200].copy()
df["item_seq_number"] = df['item_seq_number'].astype('int') // 10 * 10
plt.figure(figsize = (12, 8))
sns.barplot('item_seq_number','deal_probability', data=df)
plt.ylabel('mean_deal_probability')
plt.xlabel('item_seq_number')
plt.title('item_seq_number vs mean_deal_probability')
plt.show()
df = dtrain[dtrain.item_seq_number < 2000].copy()
df["item_seq_number"] = df['item_seq_number'].astype('int') // 100 * 100
plt.figure(figsize = (12, 8))
sns.barplot('item_seq_number','deal_probability', data=df)
plt.ylabel('mean_deal_probability')
plt.xlabel('item_seq_number')
plt.title('item_seq_number vs mean_deal_probability')
plt.show()
plt.figure(figsize = (12, 8))
df = dtrain[dtrain.item_seq_number < 20].copy()
sns.barplot('item_seq_number','price', data=df)
plt.ylabel('mean_price')
plt.xlabel('item_seq_number')
plt.title('item_seq_number vs mean_price')
plt.show()
plt.figure(figsize = (12, 8))
df = df[df.item_seq_number != 2].copy()
sns.barplot('item_seq_number','price', data=df)
plt.ylabel('mean_price')
plt.xlabel('item_seq_number')
plt.title('item_seq_number vs mean_price (w/o item_seq_number == 2)')
plt.show()
df = dtrain[dtrain.image_top_1 < 200].copy()
df['image_top_1'] = df['image_top_1'].astype('int') // 10 * 10
f, ax = plt.subplots(figsize=[12,9])
ax.set_xticklabels(ax.get_xticklabels(), rotation =90)
sns.barplot('image_top_1','deal_probability', data=df)
plt.ylabel('mean_deal_probability')
plt.xlabel('image_top_1')
plt.title('image_top_1 vs mean_deal_probability')
plt.show()
print("range of price:",np.max(dtrain.price),np.min(dtrain.price))
df = dtrain[dtrain.price < 100000].copy()
df['price'] = df['price'].astype('int') // 5000 * 5000
f, ax = plt.subplots(figsize=[12,9])
ax.set_xticklabels(ax.get_xticklabels(), rotation =90)
sns.barplot('price','deal_probability', data=df)
plt.ylabel('mean_deal_probability')
plt.xlabel('price')
plt.title('price vs mean_deal_probability')
plt.show()