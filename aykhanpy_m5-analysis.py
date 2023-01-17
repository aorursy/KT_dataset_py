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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.core.display import HTML

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
def Correlation_heat_map(df):

    ax = sns.heatmap(df.corr(), vmin = -1, vmax = 1)

    labels = [t.get_text() for t in ax.get_xticklabels()]

    ax.set_xticklabels(labels, rotation=30, horizontalalignment="right")

    sns.set(rc={'figure.figsize':(10,10)}, font_scale=1.4)
def desc(df, name):

    cm = sns.light_palette("gray", as_cmap=True)

    display(HTML('<h2><B><span style="padding-left: 30%";>' + f"{name} shape {df.shape}" + "</span></h2>"))    

    desc = pd.DataFrame(

            {

                'dtypes' : df.dtypes,

                'nunique': df.nunique(),

                "nans": df.isna().sum()

            }

        ).reset_index(level=0).merge(df.describe().T.reset_index(level=0), how = 'left').sort_values('nunique')

    style1 = desc.style.background_gradient(cmap=cm, subset=['nunique'])

    style2 = desc.style.set_properties(**{'font-weight': 'bold'}, subset = ['index'])

    style2.use(style1.export())

    display(style2)

    return desc
calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

sales_train = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
desc(calendar, 'calendar');

desc(sell_prices, 'sell_prices');

display(HTML('<h2><B><span style="padding-left: 30%";>' + f"sales_train shape {sales_train.shape}" + "</span></h2>"))

display(sales_train.head(5))
labels = ['FOODS', 'HOBBIES', 'HOUSEHOLD']

sales_by_cat = sales_train.groupby('cat_id').sum()

sales_by_cat= sales_by_cat.T.reset_index()

sales_by_cat.columns = ['d', 'FOODS', 'HOBBIES', 'HOUSEHOLD']

sales_by_cat = sales_by_cat.merge(calendar[['date', 'd']])

sales_by_cat = sales_by_cat.drop('d', axis = 1)

sales_by_cat = sales_by_cat.set_index('date')

sales_by_cat.index = pd.to_datetime(sales_by_cat.index)

sales_by_cat.resample('W').mean().plot(figsize = (20,12))

plt.title("Distribution of sales by categories in every store", fontdict={'size': 25});

display(HTML('<h2><B><span style="padding-left: 20%";>' + "Days when no store sold food or household category product" + "</span></h2>"))

calendar.loc[calendar.d.str[2:].astype('int').isin(np.where(sales_by_cat == 0)[0])]
sales_by_cat = None

labels = None
store_cat = sales_train.groupby(['store_id', 'cat_id'],as_index=True).sum()

labels = store_cat.index
c = 1

plt.figure(figsize=(23,40))

for i in store_cat.index.get_level_values(0).unique():

    plt.subplot(5,2,c)

    c +=1

    temp = store_cat.loc[i].T.reset_index()

    temp.columns = ['d', 'FOODS', 'HOBBIES', 'HOUSEHOLD']

    temp = temp.merge(calendar[['date', 'd']])

    temp = temp.drop('d', axis = 1)

    temp = temp.set_index('date')

    temp.index = pd.to_datetime(temp.index)

    plt.title(i)

    temp.resample('W').mean().plot(ax = plt.gca())

    plt.legend(loc='best')
store_gr = sales_train.groupby(['store_id'],as_index=True).sum()

store_gr = store_gr.T.reset_index()

store_gr.columns = ['d', 'CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1',

       'WI_2', 'WI_3']

store_gr = store_gr.merge(calendar[['d','date']]).drop('d',axis=1)

store_gr = store_gr.set_index('date')

store_gr.index = pd.to_datetime(store_gr.index)



plt.figure(figsize=(23,13))

plt.title('Distribution of sales by every store (two month resample)', fontdict={'size': 25})

store_gr.resample('2M').mean().plot(ax = plt.gca())

plt.xlabel('Date', fontdict = {'size': 20});

store_cat = None

store_gr = None

labels = None
plt.figure(figsize = (20,7))

plt.title('Price distribution by category', fontdict={'size':20})

plt.xlim(0,22)

sns.distplot(sell_prices.loc[sell_prices.item_id.str.contains('HOBBIES')].sell_price, label='HOBBIES')

sns.distplot(sell_prices.loc[sell_prices.item_id.str.contains('FOODS')].sell_price, label='FOODS')

sns.distplot(sell_prices.loc[sell_prices.item_id.str.contains('HOUSEHOLD')].sell_price, label='HOUSEHOLD')

plt.legend(loc = 'best')

plt.figure(figsize = (16,13))

plt.title('Price distribution by dept id', fontdict={'size':20})



plt.subplot(3,1,1)

plt.xlim(0,16)

sns.distplot(sell_prices.loc[sell_prices.item_id.str.contains('HOBBIES_1')].sell_price, label='HOBBIES_1')

sns.distplot(sell_prices.loc[sell_prices.item_id.str.contains('HOBBIES_2')].sell_price, label='HOBBIES_2')

plt.xlabel('')



plt.legend(loc = 'best')





plt.subplot(3,1,2)

plt.xlim(0,22)

sns.distplot(sell_prices.loc[sell_prices.item_id.str.contains('HOUSEHOLD_1')].sell_price, label='HOUSEHOLD_1')

sns.distplot(sell_prices.loc[sell_prices.item_id.str.contains('HOUSEHOLD_2')].sell_price, label='HOUSEHOLD_2')

plt.xlabel('')

plt.legend(loc = 'best')





plt.subplot(3,1,3)

plt.xlim(0,13)

sns.distplot(sell_prices.loc[sell_prices.item_id.str.contains('FOODS_1')].sell_price, label='FOODS_1')

sns.distplot(sell_prices.loc[sell_prices.item_id.str.contains('FOODS_2')].sell_price, label='FOODS_2')

sns.distplot(sell_prices.loc[sell_prices.item_id.str.contains('FOODS_3')].sell_price, label='FOODS_3')

plt.xlabel('Sell Price', fontdict={'size': 13})

plt.legend(loc = 'best')
