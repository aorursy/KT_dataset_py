# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic



import statsmodels.api as sm



import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv', parse_dates=['date'])

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')
train = pd.merge(train, items.drop('item_name', axis=1),on='item_id') #train.join(items.drop('item_name', axis=1), on=['item_id'])

train.sample(5)
categories.head(3)
def build_dict(o,t):

    l = []

    s,f = 0,0

    c = ""

    while f < len(t):

        while f < len(t) and not t[f].isalpha() :

            f += 1

        while f < len(t) and t[f] != "\'" :

            c += t[f]

            f += 1

        if c != "":

            l.append(c)

        c = ""

#     print(*zip(o,l), sep="\n")

    dic = dict(zip(o,l))

    return dic



t = '''

'PC - Headsets / Headphones' 'Accessories - PS2' 'Accessories - PS3'

 'Accessories - PS4' 'Accessories - PSP' 'Accessories - PSVita'

 'Accessories - XBOX 360' 'Accessories - XBOX ONE' 'Tickets (Digit)'

 '' Product delivery '' Game consoles - PS2 '' Game consoles - PS3 '

 'Game consoles - PS4' 'Game consoles - PSP'

 'Game consoles - PSVita' 'Game consoles - XBOX 360'

 'Game consoles - XBOX ONE' 'Game consoles - Other' 'Games - PS2'

 '' Games - PS3 '' Games - PS4 '' Games - PSP '' Games - PSVita '' Games - XBOX 360 '

 '' Games - XBOX ONE '' Games - Accessories for games' 'Android Games - Digit'

 'MAC Games - Digital' PC Games - Additional Editions'

 'PC Games - Collector's Editions' 'PC Games - Standard Editions'

 'PC Games - Digit' 'Payment cards (Cinema, Music, Games)'

 'Payment Cards - Live!' 'Payment Cards - Live! (Numeral)'

 'Payment Cards - PSN' 'Payment Cards - Windows (Digital)' 'Cinema - Blu-Ray'

 'Cinema - Blu-Ray 3D' 'Cinema - Blu-Ray 4K' 'Cinema - DVD'

 'Cinema - Collection' 'Books - Artbooks, Encyclopedias'

 'Books - Audiobooks' 'Books - Audiobooks (Digit)' 'Books - Audiobooks 1C'

 'Books - Business Literature' 'Books - Comics, Manga'

 'Books - Computer literature' 'Books - Methodological materials 1C'

 'Books - Postcards' 'Books - Cognitive literature'

 'Books - Guides' 'Books - Fiction'

 'Books - Digital' 'Music - Local Production CD'

 'Music - Corporate Production CD' 'Music - MP3' 'Music - Vinyl'

 'Music - Music Video' 'Music - Gift Editions'

 'Gifts - Attributes' 'Gifts - Gadgets, robots, sports'

 'Gifts - Soft Toys' 'Gifts - Board Games'

 'Gifts - Souvenirs (in bulk)'

 'Gifts - Bags, Albums, Mousepads' 'Gifts - Figures'

 'Programs - 1C: Enterprise 8' 'Programs - MAC (Digit)'

 'Programs - For home and office' 'Programs - For home and office (Digit)'

 'Programs - Educational' 'Programs - Educational (Numeral)' 'Utilities'

 'Service - Tickets' 'Net carriers (spire)'

 'Clean media (piece)' 'Batteries'

'''



dic = build_dict(list(categories.item_category_name.unique()), t)

categories.replace(dic, inplace=True)

categories.sample(5)
print(train.shape, test.shape)

print("train has all shops in test? ",set(test.shop_id.unique()).issubset(set(train.shop_id.unique())))

print("train has all items in test? ",set(test.item_id.unique()).issubset(set(train.item_id.unique())))

print("number of items in test is: ", len(set(test.item_id.unique())))

print("how many items in test are not in train? ", len(set(test.item_id.unique()).difference(set(train.item_id.unique()))) )

print("percent missing: ", len(set(test.item_id.unique()).difference(set(train.item_id.unique())))/len(set(test.item_id.unique()))*100)
ts=train.groupby(["date_block_num"])["item_cnt_day"].sum()

ts.astype('float')

plt.figure(figsize=(16,8))

plt.title('Total Sales of the company')

plt.xlabel('Time (Month)')

plt.ylabel('Sales (Item Count)')

plt.plot(ts);
res = sm.tsa.seasonal_decompose(ts.values,freq=12) #,model="multiplicative")

res.plot();
mod = sm.tsa.statespace.SARIMAX(ts,

                                order=(1, 1, 1),

                                seasonal_order=(1, 1, 0, 12),

                                enforce_stationarity=False,

                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])
results.plot_diagnostics(lags=4, figsize=(16, 8)) #lags default at 10 is too much and cause acf plot error

plt.show()
pred = results.get_prediction(start=28, dynamic=False)

pred_ci = pred.conf_int()

ax = ts.plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Time (Month)')

ax.set_ylabel('Sales (Item Count)')

plt.legend()

plt.show()
y_forecasted = pred.predicted_mean

y_truth = ts[28:]

mse = ((y_forecasted - y_truth) ** 2).mean()

print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
pred_uc = results.get_forecast(steps=5)

pred_ci = pred_uc.conf_int()

ax = ts.plot(label='observed', figsize=(14, 7))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Time (Month)')

ax.set_ylabel('Sales (Item Count)')

plt.legend()

plt.show()
print(pred_ci,'\n', pred_ci.mean(axis=1))
predicted_sale = pred_ci.iloc[0].mean()

last_month_sale = ts.iloc[-1]

last_year_sale = ts.iloc[-12]

print(last_month_sale, last_year_sale, predicted_sale)

rlm = predicted_sale/last_month_sale

rly = predicted_sale/last_year_sale

print(rlm, rly)
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

print('test shape before ', test.shape)

test = pd.merge(test, items.drop('item_name', axis=1),on='item_id') #train.join(items.drop('item_name', axis=1), on=['item_id'])

print('test shape after ', test.shape)

test.head(5)
monthly_sale_shop_item = train.groupby(["shop_id","item_id","date_block_num"])["item_cnt_day"].sum()

monthly_sale_shop_item.head()
monthly_sale_shop_item = monthly_sale_shop_item.reset_index()

monthly_sale_shop_item.head()
last_month_shop_item = monthly_sale_shop_item[monthly_sale_shop_item['date_block_num'] == 33]

last_month_shop_item.head()
last_year_shop_item = monthly_sale_shop_item[monthly_sale_shop_item['date_block_num'] == 22]

last_year_shop_item.head()
ttest = pd.merge(test, last_month_shop_item.drop('date_block_num', axis=1),on=['shop_id','item_id'], how='left')

print(ttest.shape)

ttest.head(5)
ttest = pd.merge(ttest, last_year_shop_item.drop('date_block_num', axis=1),on=['shop_id','item_id'], suffixes=('_33', '_22'), how='left') 

print(ttest.shape)

ttest.head(5)
ttest.isnull().sum()
cat_sale = ttest.groupby(['shop_id', 'item_category_id'])['item_cnt_day_33'].apply(lambda x: x.mode(dropna=True))

cat_sale = cat_sale.reset_index()

print(cat_sale.shape)

cat_sale.head()
cat_sale.rename(columns={'level_2':'shop_cat_mode'}, 

                 inplace=True)

cat_sale.head()
print(ttest.shape)

ttest = pd.merge(ttest, cat_sale.drop('item_cnt_day_33', axis=1),on=['shop_id','item_category_id'], how='left') 

print(ttest.shape)

ttest.head(5)
ttest.isnull().sum()
ttest['shop_cat_mode'].value_counts(dropna=False)