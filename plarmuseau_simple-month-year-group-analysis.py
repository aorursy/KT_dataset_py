import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
df = pd.read_csv('../input/vb01.csv' ,names=['atcnr','atc','doping','ddd','dpp','ehd','tva','nr','prdnr','code','aant','pp','salesnr','client','date'] )
df=df[df['date']>'2005-06-19']
df
df['atc3']=df['atc'].str[:3]
df['atc5']=df['atc'].str[:5]
df['tel']=1.0
df['dpp']=pd.Series(df['dpp'].replace(',','.'),dtype='float')
df
# Expand dataframe with more useful columns
def expand_df(dfi):
    data = dfi.copy()
    data['day'] = pd.DatetimeIndex(data.date).day
    data['month'] = pd.DatetimeIndex(data.date).month
    data['year'] = pd.DatetimeIndex(data.date).year
    data['dayofweek'] = pd.DatetimeIndex(data.date).dayofweek
    return data

df = expand_df(df[:-2])
display(df)




groep=pd.pivot_table(df, index=['year','month'],columns='atc3',values=['tel'],aggfunc=np.sum)
#pd.plotting.scatter_matrix(groep, alpha = 0.3, figsize = (14,8), diagonal = 'kde')
groep[1:].plot()

import matplotlib.pyplot as plt

agg_year_item = pd.pivot_table(df, index='year', columns='atc3',
                               values='tel', aggfunc=np.sum).values
agg_year_store = pd.pivot_table(df, index='year', columns='atc3',
                                values='aant', aggfunc=np.sum).values

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(agg_year_item / agg_year_item.mean(0)[np.newaxis])
plt.title("Items")
plt.xlabel("Year")
plt.ylabel("Relative Sales")
plt.subplot(122)
plt.plot(agg_year_store / agg_year_store.mean(0)[np.newaxis])
plt.title("Stores")
plt.xlabel("Year")
plt.ylabel("Relative Sales")
plt.show()
agg_month_item = pd.pivot_table(df, index='month', columns='atc3',
                                values='tel', aggfunc=np.sum).values
agg_month_store = pd.pivot_table(df, index='month', columns='atc3',
                                 values='aant', aggfunc=np.sum).values

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(agg_month_item[1:] / agg_month_item.mean(0)[np.newaxis])
plt.title("Items")
plt.xlabel("Month")
plt.ylabel("Relative Sales")
plt.subplot(122)
plt.plot(agg_month_store[1:] / agg_month_store.mean(0)[np.newaxis])
plt.title("Stores")
plt.xlabel("Month")
plt.ylabel("Relative Sales")
plt.show()
agg_dow_item = pd.pivot_table(df, index='dayofweek', columns='atc3',
                              values='tel', aggfunc=np.sum).values
agg_dow_store = pd.pivot_table(df, index='dayofweek', columns='atc3',
                               values='aant', aggfunc=np.mean).values

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(agg_dow_item / agg_dow_item.mean(0)[np.newaxis])
plt.title("Items")
plt.xlabel("Day of Week")
plt.ylabel("Relative Sales")
plt.subplot(122)
plt.plot(agg_dow_store / agg_dow_store.mean(0)[np.newaxis])
plt.title("Stores")
plt.xlabel("Day of Week")
plt.ylabel("Relative Sales")
plt.show()
agg_dow_month = pd.pivot_table(df, index='dayofweek', columns='month',
                               values='tel', aggfunc=np.sum).values
agg_month_year = pd.pivot_table(df, index='month', columns='year',
                                values='tel', aggfunc=np.sum).values
agg_dow_year = pd.pivot_table(df, index='dayofweek', columns='year',
                              values='tel', aggfunc=np.sum).values

plt.figure(figsize=(18, 5))
plt.subplot(131)
plt.plot(agg_dow_month / agg_dow_month.mean(0)[np.newaxis])
plt.title("Months")
plt.xlabel("Day of Week")
plt.ylabel("Relative Sales")
plt.subplot(132)
plt.plot(agg_month_year / agg_month_year.mean(0)[np.newaxis])
plt.title("Years")
plt.xlabel("Months")
plt.ylabel("Relative Sales")
plt.subplot(133)
plt.plot(agg_dow_year / agg_dow_year.mean(0)[np.newaxis])
plt.title("Years")
plt.xlabel("Day of Week")
plt.ylabel("Relative Sales")
plt.show()
grand_avg = df.tel.mean()
# Item-Store Look Up Table
store_item_table = pd.pivot_table(df, index='code', columns='atc3',
                                  values='tel', aggfunc=np.sum)
display(store_item_table)

# Monthly pattern
month_table = pd.pivot_table(df, index='month', values='tel', aggfunc=np.sum)
month_table.tel /= grand_avg

# Day of week pattern
dow_table = pd.pivot_table(df, index='dayofweek', values='tel', aggfunc=np.sum)
dow_table.tel /= grand_avg

# Yearly growth pattern
year_table = pd.pivot_table(df, index='year', values='tel', aggfunc=np.sum)
year_table /= grand_avg

years = np.arange(2005, 2019)
annual_sales_avg = year_table.values.squeeze()

p1 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 1))
p2 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 2))

plt.figure(figsize=(8,6))
plt.plot(years[:-1], annual_sales_avg, 'ko')
plt.plot(years, p1(years), 'C0-')
plt.plot(years, p2(years), 'C1-')
plt.xlim(2005, 2018.5)
plt.title("Relative Sales by Year")
plt.ylabel("Relative Sales")
plt.xlabel("Year")
plt.show()

print(f"2018 Relative Sales by Degree-1 (Linear) Fit = {p1(2018):.4f}")
print(f"2018 Relative Sales by Degree-2 (Quadratic) Fit = {p2(2018):.4f}")

# We pick the quadratic fit
annual_growth = p2