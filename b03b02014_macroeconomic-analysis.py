# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

# first let's average per day

gb = train.groupby(['timestamp'])

gb.sum().head()

dfagg = pd.DataFrame()

dfagg['avg_price_per_sqm'] = gb.price_doc.sum() / gb.full_sq.sum()

dfagg['rolling_average_immo'] = dfagg['avg_price_per_sqm'].rolling(30).mean()

dfagg.reset_index(inplace=True)



macro_df = pd.read_csv("../input/macro.csv")

macro_df['date'] = pd.to_datetime(macro_df['timestamp'])

#macro_df['month'] = macro_df['month'].month

macro_df.head(1)



dfagg = pd.merge(dfagg, macro_df, how='left', on=['timestamp'])

dfagg.head(1)
plt.figure(figsize=(12,6))

plt.plot(dfagg['date'], dfagg['gdp_quart'], label='gdp')

plt.plot(dfagg['date'], dfagg['rolling_average_immo']/5, label='real estate')

plt.title('Real estate price per square meter vs GDP')

plt.legend(loc='lower right')



plt.ylim(0, 32000)

plt.show()
plt.figure(figsize=(12,6))

plt.plot(dfagg['date'], dfagg['gdp_quart_growth']*5, label='gdp_quart_growth')

plt.plot(dfagg['date'], dfagg['rolling_average_immo']/10000, label='real estate')

plt.title('Real estate price per square meter vs GDP')

plt.legend(loc='lower right')



#plt.ylim(0, 32000)

plt.show()
plt.figure(figsize=(12,6))

plt.plot(macro_df['date'], macro_df['micex'], label='micex')

plt.plot(dfagg['date'], dfagg['rolling_average_immo']/90, label='real estate')

plt.title('Real estate price per square meter vs micex')

plt.legend(loc='lower right')

plt.ylim(0, 2200)

plt.show()
plt.figure(figsize=(12,6))

plt.plot(macro_df['date'], macro_df['micex_cbi_tr']*5, label='micex_cbi_tr')

plt.plot(dfagg['date'], dfagg['rolling_average_immo']/90, label='real estate')

plt.title('Real estate price per square meter vs micex_cbi_tr')

plt.legend(loc='lower right')

plt.ylim(0, 2200)

plt.show()
plt.figure(figsize=(12,6))

plt.plot(macro_df['date'], macro_df['usdrub'], label='usdrub')

plt.plot(macro_df['date'], macro_df['eurrub'], label='eurrub')

plt.plot(dfagg['date'], dfagg['rolling_average_immo']/2000, label='real estate', linewidth=3)

plt.title('Rolling average price per square meter vs Exchange rates')

plt.legend(loc='lower right')

plt.ylim(0, 100)

plt.show()
plt.figure(figsize=(12,6))

plt.plot(macro_df['date'], macro_df['mortgage_rate'], label='mortgage_rate')

plt.plot(macro_df['date'], macro_df['deposits_rate'], label='deposits_rate')

plt.plot(dfagg['date'], dfagg['rolling_average_immo']/10000, label='real estate', linewidth=3)

plt.title('Rolling average price per square meter vs mortgage rates, deposits rates')

plt.legend(loc='lower right')

plt.ylim(0, 20)

plt.show()
plt.figure(figsize=(12,6))

plt.plot(macro_df['date'], macro_df['oil_urals']*1000, label='oil_urals')

plt.plot(macro_df['date'], macro_df['micex']*100, label='micex')

plt.plot(dfagg['date'], dfagg['rolling_average_immo'], label='real estate', linewidth=3)

plt.title('Rolling average price per square meter vs Micex, and Oil price')

plt.legend(loc='lower right')

plt.ylim(0, 220000)

plt.show()
gb = train[train['kremlin_km']<=20].groupby(['timestamp'])

gb.sum().head()

dfagg = pd.DataFrame()

dfagg['avg_price_per_sqm'] = gb.price_doc.sum() / gb.full_sq.sum()

dfagg['rolling_average_immo'] = dfagg['avg_price_per_sqm'].rolling(30).mean()

dfagg.reset_index(inplace=True)



macro_df = pd.read_csv("../input/macro.csv")

macro_df['date'] = pd.to_datetime(macro_df['timestamp'])

#macro_df['month'] = macro_df['month'].month

macro_df.head(1)



dfagg = pd.merge(dfagg, macro_df, how='left', on=['timestamp'])

dfagg.head(1)
plt.figure(figsize=(12,6))

plt.plot(macro_df['date'], macro_df['micex'], label='micex')

plt.plot(dfagg['date'], dfagg['rolling_average_immo']/90, label='real estate')

plt.title('Moscow - Real estate price per square meter vs micex')

plt.legend(loc='lower right')

plt.ylim(0, 2200)

plt.show()
train['dist'] = np.round(train['kremlin_km']/5)
train.groupby(['dist']).count()
# first let's average per day

train['date'] = pd.to_datetime(train['timestamp'])



gb = train.groupby(['dist', 'date'])

gb.sum().head()

dfagg = pd.DataFrame()



dfagg['avg_price_per_sqm'] = gb.price_doc.sum() / gb.full_sq.sum()

dfagg.reset_index(inplace=True)

dfagg['rolling_average_immo_1'] = dfagg[dfagg['dist']==1]['avg_price_per_sqm'].rolling(30).mean()

dfagg['rolling_average_immo_2'] = dfagg[dfagg['dist']==2]['avg_price_per_sqm'].rolling(30).mean()

dfagg['rolling_average_immo_3'] = dfagg[dfagg['dist']==3]['avg_price_per_sqm'].rolling(30).mean()

dfagg['rolling_average_immo_4'] = dfagg[dfagg['dist']==4]['avg_price_per_sqm'].rolling(30).mean()



dfagg.head(1)
plt.figure(figsize=(14,8))

#plt.plot(dfagg[dfagg['dist']==0]['avg_price_per_sqm'], label='avg price per square meter')

plt.plot(dfagg['date'], dfagg['rolling_average_immo_1'], label='avg price per square meter - 5km')

plt.plot(dfagg['date'], dfagg['rolling_average_immo_2'], label='avg price per square meter - 10km')

plt.plot(dfagg['date'], dfagg['rolling_average_immo_3'], label='avg price per square meter - 15km')

plt.plot(dfagg['date'], dfagg['rolling_average_immo_4'], label='avg price per square meter - 20km')

plt.title('Average price per square meter depending on distance to Kremlin')

plt.xlabel('days')

plt.ylabel('average price per full_sqm')

plt.legend(loc='lower right')

plt.ylim(30000, 240000)

plt.show()