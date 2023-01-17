import pandas as pd 

import numpy as np 

import seaborn as sns

import scipy

import matplotlib.pyplot as plt

sns.set_style('darkgrid')

%matplotlib inline
cat=pd.read_csv('../input/competitive-data-science-final-project/sales_train.csv/item_categories.csv')

item=pd.read_csv('../input/competitive-data-science-final-project/sales_train.csv/items.csv')

train=pd.read_csv('../input/competitive-data-science-final-project/sales_train.csv/sales_train.csv')

submission=pd.read_csv('../input/competitive-data-science-final-project/sales_train.csv/sales_train.csv')

shops=pd.read_csv('../input/competitive-data-science-final-project/sales_train.csv/shops.csv')

test=pd.read_csv('../input/competitive-data-science-final-project/test.csv/test.csv')

cat.head()
item.head()
train.head()
shops.head()
test.head()
train=pd.merge(train,item,on='item_id',how='inner')

train=pd.merge(train,cat,on='item_category_id',how='inner')

train=pd.merge(train,shops,on='shop_id',how='inner')



test=pd.merge(test,item,on='item_id',how='inner')

test=pd.merge(test,cat,on='item_category_id',how='inner')

test=pd.merge(test,shops,on='shop_id',how='inner')
train.head()
test.head()
train.describe()
train=train[(train['item_price'] > 0) & (train['item_cnt_day'] >= 0)]
dict_categories = ['Cinema - DVD', 'PC Games - Standard Editions',

                    'Music - Local Production CD', 'Games - PS3', 'Cinema - Blu-Ray',

                    'Games - XBOX 360', 'PC Games - Additional Editions', 'Games - PS4',

                    'Gifts - Stuffed Toys', 'Gifts - Board Games (Compact)',

                    'Gifts - Figures', 'Cinema - Blu-Ray 3D',

                    'Programs - Home and Office', 'Gifts - Development',

                    'Gifts - Board Games', 'Gifts - Souvenirs (on the hinge)',

                    'Cinema - Collection', 'Music - MP3', 'Games - PSP',

                    'Gifts - Bags, Albums, Mouse Pads', 'Gifts - Souvenirs',

                    'Books - Audiobooks', 'Gifts - Gadgets, robots, sports',

                    'Accessories - PS4', 'Games - PSVita',

                    'Books - Methodical materials 1C', 'Payment cards - PSN',

                    'PC Games - Digit', 'Games - Game Accessories', 'Accessories - XBOX 360',

                    'Accessories - PS3', 'Games - XBOX ONE', 'Music - Vinyl',

                    'Programs - 1C: Enterprise 8', 'PC Games - Collectible Editions',

                    'Gifts - Attributes', 'Service Tools',

                    'Music - branded production CD', 'Payment cards - Live!',

                    'Game consoles - PS4', 'Accessories - PSVita', 'Batteries',

                    'Music - Music Video', 'Game Consoles - PS3',

                    'Books - Comics, Manga', 'Game Consoles - XBOX 360',

                    'Books - Audiobooks 1C', 'Books - Digit',

                    'Payment cards (Cinema, Music, Games)', 'Gifts - Cards, stickers',

                    'Accessories - XBOX ONE', 'Pure media (piece)',

                    'Programs - Home and Office (Digital)', 'Programs - Educational',

                    'Game consoles - PSVita', 'Books - Artbooks, encyclopedias',

                    'Programs - Educational (Digit)', 'Accessories - PSP',

                    'Gaming consoles - XBOX ONE', 'Delivery of goods',

                    'Payment Cards - Live! (Figure) ',' Tickets (Figure) ',

                    'Music - Gift Edition', 'Service Tools - Tickets',

                    'Net media (spire)', 'Cinema - Blu-Ray 4K', 'Game consoles - PSP',

                    'Game Consoles - Others', 'Books - Audiobooks (Figure)',

                    'Gifts - Certificates, Services', 'Android Games - Digit',

                    'Programs - MAC (Digit)', 'Payment Cards - Windows (Digit)',

                    'Books - Business Literature', 'Games - PS2', 'MAC Games - Digit',

                    'Books - Computer Literature', 'Books - Travel Guides',

                    'PC - Headsets / Headphones', 'Books - Fiction',

                    'Books - Cards', 'Accessories - PS2', 'Game consoles - PS2',

                    'Books - Cognitive literature']



dict_shops = ['Moscow Shopping Center "Semenovskiy"', 

              'Moscow TRK "Atrium"', 

              "Khimki Shopping Center",

              'Moscow TC "MEGA Teply Stan" II', 

              'Yakutsk Ordzhonikidze, 56',

              'St. Petersburg TC "Nevsky Center"', 

              'Moscow TC "MEGA Belaya Dacha II"',

              'Voronezh (Plekhanovskaya, 13)', 

              'Yakutsk Shopping Center "Central"',

              'Chekhov SEC "Carnival"', 

              'Sergiev Posad TC "7Ya"',

              'Tyumen TC "Goodwin"',

              'Kursk TC "Pushkinsky"', 

              'Kaluga SEC "XXI Century"',

              'N.Novgorod Science and entertainment complex "Fantastic"',

              'Moscow MTRC "Afi Mall"',

              'Voronezh SEC "Maksimir"', 'Surgut SEC "City Mall"',

              'Moscow Shopping Center "Areal" (Belyaevo)', 'Krasnoyarsk Shopping Center "June"',

              'Moscow TK "Budenovsky" (pav.K7)', 'Ufa "Family" 2',

              'Kolomna Shopping Center "Rio"', 'Moscow Shopping Center "Perlovsky"',

              'Moscow Shopping Center "New Century" (Novokosino)', 'Omsk Shopping Center "Mega"',

              'Moscow Shop C21', 'Tyumen Shopping Center "Green Coast"',

              'Ufa TC "Central"', 'Yaroslavl shopping center "Altair"',

              'RostovNaDonu "Mega" Shopping Center', '"Novosibirsk Mega "Shopping Center',

              'Samara Shopping Center "Melody"', 'St. Petersburg TC "Sennaya"',

              "Volzhsky Shopping Center 'Volga Mall' ",

              'Vologda Mall "Marmelad"', 'Kazan TC "ParkHouse" II',

              'Samara Shopping Center ParkHouse', '1C-Online Digital Warehouse',

              'Online store of emergencies', 'Adygea Shopping Center "Mega"',

              'Balashikha shopping center "October-Kinomir"' , 'Krasnoyarsk Shopping center "Vzletka Plaza" ',

              'Tomsk SEC "Emerald City"', 'Zhukovsky st. Chkalov 39m? ',

              'Kazan Shopping Center "Behetle"', 'Tyumen SEC "Crystal"',

              'RostovNaDonu TRK "Megacenter Horizon"',

              '! Yakutsk Ordzhonikidze, 56 fran', 'Moscow TC "Silver House"',

              'Moscow TK "Budenovsky" (pav.A2)', "N.Novgorod SEC 'RIO' ",

              '! Yakutsk TTS "Central" fran', 'Mytishchi TRK "XL-3"',

              'RostovNaDonu TRK "Megatsentr Horizon" Ostrovnoy', 'Exit Trade',

              'Voronezh SEC City-Park "Grad"', "Moscow 'Sale'",

              'Zhukovsky st. Chkalov 39m² ',' Novosibirsk Shopping Mall "Gallery Novosibirsk"']



train.item_category_name = train.item_category_name.map(dict(zip(train.item_category_name.value_counts().index, dict_categories)))

train.shop_name = train.shop_name.map(dict(zip(train.shop_name.value_counts().index, dict_shops)))
#Shop Analysis

#Based on price

tst=train.groupby('shop_name')['item_price'].sum().reset_index().set_index('shop_name').sort_values('item_price',ascending=False)

tst.head(10).plot(kind='bar')

plt.title('Top 10 Shops with Highest Items Price')

plt.xlabel('Shop Name')

plt.ylabel('Total Price')
train['revenue_day']=train['item_cnt_day']*train['item_price']

#test['revenue_day']=test['item_cnt_day']*test['item_price']
#Based on revenue

tst=train.groupby('shop_name')['revenue_day'].sum().reset_index().set_index('shop_name').sort_values('revenue_day',ascending=False)

tst.head(10).plot(kind='bar')

plt.title('Top 10 Shops by Their Total Revenue')

plt.xlabel('Shop Name')

plt.ylabel('Total Revenue')
#Based on sold items

tst=train.groupby('item_name')['item_cnt_day'].sum().reset_index().set_index('item_name').sort_values('item_cnt_day',ascending=False)

tst.head(10).plot(kind='bar')

plt.title('Top 10 Sold Items Amount')

plt.xlabel('Item Name')

plt.ylabel('Total Amount of Sold Items')
#Based on sold items

tst=train.groupby('item_category_name')['item_cnt_day'].sum().reset_index().set_index('item_category_name').sort_values('item_cnt_day',ascending=False)

tst.head(10).plot(kind='bar')

plt.title('Top 10 Sold Items Amount (Category-Based)')

plt.xlabel('Item Category')

plt.ylabel('Total Amount of Sold Items')
test
#Time Series

train['date']=pd.to_datetime(train.date)

#test['date']=pd.to_datetime(test.date,format='%d.%M.%Y')
train['day']=pd.DatetimeIndex(train['date']).day

train['month']=pd.DatetimeIndex(train['date']).month

train['year']=pd.DatetimeIndex(train['date']).year
train.head()
train=train.sort_values(by='date',ascending=True)
tst=train.groupby(['date'])['item_cnt_day'].sum().to_frame().reset_index()
tst['date']
test
#Time Analysis

plt.figure(figsize=(16,8))

train.groupby('date_block_num')['item_cnt_day'].sum().plot(kind='line')

plt.title('Total Sales of The Company')

plt.show()
plt.figure(figsize=(16,8))

plt.plot(train.groupby('date_block_num')['item_cnt_day'].sum().rolling(window=12,center=False).mean(),label="Rolling Mean")

plt.title('Yearly Sales Trend')

plt.xlabel('n-th Month')

plt.ylabel('Sales')

plt.show()
#Check Seasonality, Trends,etc

import statsmodels.api as sm

res = sm.tsa.seasonal_decompose(train.groupby('date_block_num')['item_cnt_day'].sum().values,freq=12)

fig=res.plot()

plt.show()
def cross_heatmap(df, cols, normalize=False, values=None, aggfunc=None):

    temp = cols

    cm = sns.light_palette("green", as_cmap=True)

    return pd.crosstab(df[temp[0]], df[temp[1]], 

                       normalize=normalize, values=values, aggfunc=aggfunc).style.background_gradient(cmap = cm)
train['revenue']=train['item_cnt_day']*train['item_price']
cross_heatmap(train, ['item_category_name', 'shop_name'], 

              normalize='columns', aggfunc='sum', values=train['revenue'])
cross_heatmap(train, ['item_category_name', 'shop_name'], 

              normalize='columns', aggfunc='sum', values=train['item_cnt_day'])
monthly_df=train.groupby(['shop_name','date_block_num'])['item_cnt_day'].sum().to_frame()
monthly_df=monthly_df.reset_index()
monthly_df.describe()
train.describe()
monthly_df=train.groupby(['item_id','month'])['item_cnt_day'].sum().reset_index()

upper_bound=1.5*13+15

outlier=monthly_df[monthly_df['item_cnt_day'] > upper_bound]

print('Outlier percentage: {}% of data'.format(outlier.shape[0]/monthly_df.shape[0]))
monthly_df=train.groupby(['item_id','month'])['item_cnt_day'].sum().reset_index()

upper_bound=monthly_df['item_cnt_day'].quantile(0.75)+1.5*(monthly_df['item_cnt_day'].quantile(0.75)-monthly_df['item_cnt_day'].quantile(0.25))

lower_bound=monthly_df['item_cnt_day'].quantile(0.25)-1.5*(monthly_df['item_cnt_day'].quantile(0.75)-monthly_df['item_cnt_day'].quantile(0.25))

outlier=monthly_df[(monthly_df['item_cnt_day'] > upper_bound) | (monthly_df['item_cnt_day'] < lower_bound)]

print('Outlier percentage: {}% of data'.format(outlier.shape[0]/monthly_df.shape[0]))
monthly_df=train.groupby(['shop_id','month'])['item_cnt_day'].sum().reset_index()

upper_bound=monthly_df['item_cnt_day'].quantile(0.75)+1.5*(monthly_df['item_cnt_day'].quantile(0.75)-monthly_df['item_cnt_day'].quantile(0.25))

lower_bound=monthly_df['item_cnt_day'].quantile(0.25)-1.5*(monthly_df['item_cnt_day'].quantile(0.75)-monthly_df['item_cnt_day'].quantile(0.25))

outlier=monthly_df[(monthly_df['item_cnt_day'] > upper_bound) | (monthly_df['item_cnt_day'] < lower_bound)]

print('Outlier percentage: {}% of data'.format(outlier.shape[0]/monthly_df.shape[0]))
monthly_df.shape[0]