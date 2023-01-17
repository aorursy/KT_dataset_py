import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
wind=pd.read_csv('../input/ninja_wind_europe_v1.1_current_on-offshore.csv')
sun=pd.read_csv('../input/ninja_pv_europe_v1.1_sarah.csv')
# Expand dataframe with more useful columns
def expand_df(dfi):
    data = dfi.copy()
    data['day'] = pd.DatetimeIndex(data.time).day
    data['month'] = pd.DatetimeIndex(data.time).month
    data['year'] = pd.DatetimeIndex(data.time).year
    data['dayofweek'] = pd.DatetimeIndex(data.time).dayofweek
    data['hour']= pd.DatetimeIndex(data.time).hour
    return data

windbe=expand_df( wind[['time','BE_OFF','BE_ON']] )
sun=expand_df( sun[['time','BE']])
print('wind capacity factor BE',windbe['BE_OFF'].mean(),windbe['BE_ON'].mean() )

windnl=expand_df( wind[['time','NL_OFF','NL_ON']] )
print('wind capacity factor NL', wind['NL_OFF'].mean(),wind['NL_ON'].mean() )

windde=expand_df( wind[['time','DE_OFF','DE_ON']] )
print('wind capacity factor DE', wind['DE_OFF'].mean(),wind['DE_ON'].mean() )

wind=windbe
import matplotlib.pyplot as plt

agg_year_item = pd.pivot_table(wind, index='month', columns='year',
                               values='BE_OFF', aggfunc=np.mean).values
agg_year_store = pd.pivot_table(wind, index='month', columns='year',
                                values='BE_ON', aggfunc=np.mean).values

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(agg_year_item / agg_year_item.mean(0)[np.newaxis])
plt.title("off shore")
plt.xlabel("month")
plt.ylabel("avg speed")
plt.subplot(122)
plt.plot(agg_year_store / agg_year_store.mean(0)[np.newaxis])
plt.title("on_shore")
plt.xlabel("month")
plt.ylabel("avg speed")
plt.show()
import matplotlib.pyplot as plt

agg_year_item = pd.pivot_table(wind, index='hour', columns='year',
                               values='BE_OFF', aggfunc=np.mean).values
agg_year_store = pd.pivot_table(wind, index='hour', columns='year',
                                values='BE_ON', aggfunc=np.mean).values

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(agg_year_item / agg_year_item.mean(0)[np.newaxis])
plt.title("off shore")
plt.xlabel("hour")
plt.ylabel("avg speed")
plt.subplot(122)
plt.plot(agg_year_store / agg_year_store.mean(0)[np.newaxis])
plt.title("on_shore")
plt.xlabel("hour")
plt.ylabel("avg speed")
plt.show()
import matplotlib.pyplot as plt

agg_year_item = pd.pivot_table(sun, index='hour', columns='year',
                               values='BE', aggfunc=np.mean).values
agg_year_store = pd.pivot_table(sun, index='month', columns='year',
                                values='BE', aggfunc=np.mean).values

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(agg_year_item / agg_year_item.mean(0)[np.newaxis])
plt.title("sun")
plt.xlabel("hour")
plt.ylabel("avg speed")
plt.subplot(122)
plt.plot(agg_year_store / agg_year_store.mean(0)[np.newaxis])
plt.title("season ")
plt.xlabel("month")
plt.ylabel("avg speed")
plt.show()
grand_avg = sun.BE.mean()
# Item-Store Look Up Table
store_item_table = pd.pivot_table(sun, index='hour', columns='year',
                                  values='BE', aggfunc=np.sum)
display(store_item_table)

# Monthly pattern
month_table = pd.pivot_table(sun, index='month', values='BE', aggfunc=np.sum)
month_table.BE /= grand_avg

# Day of week pattern
dow_table = pd.pivot_table(sun, index='dayofweek', values='BE', aggfunc=np.sum)
dow_table.BE /= grand_avg

# Yearly growth pattern
year_table = pd.pivot_table(sun, index='year', values='BE', aggfunc=np.sum)
year_table /= grand_avg

years = np.arange(1985, 2017)
annual_sales_avg = year_table.values.squeeze()
print(len( annual_sales_avg), len(years) )
p1 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 1))
p2 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 2))

plt.figure(figsize=(8,6))
plt.plot(years[:-1], annual_sales_avg, 'ko')
plt.plot(years, p1(years), 'C0-')
plt.plot(years, p2(years), 'C1-')
plt.xlim(1985, 2018.5)
plt.title("Relative sun by Year")
plt.ylabel("Relative sun")
plt.xlabel("Year")
plt.show()

print(f"2016 Relative sun by Degree-1 (Linear) Fit = {p1(2018):.4f}")
print(f"2016 Relative sun by Degree-2 (Quadratic) Fit = {p2(2018):.4f}")

# We pick the quadratic fit
annual_growth = p2
grand_avg = wind.BE_ON.mean()
# Item-Store Look Up Table
store_item_table = pd.pivot_table(wind, index='hour', columns='year',
                                  values='BE_ON', aggfunc=np.sum)
display(store_item_table)

# Monthly pattern
month_table = pd.pivot_table(wind, index='month', values='BE_ON', aggfunc=np.sum)
month_table.BE_ON /= grand_avg

# Day of week pattern
dow_table = pd.pivot_table(wind, index='dayofweek', values='BE_ON', aggfunc=np.sum)
dow_table.BE_ON /= grand_avg

# Yearly growth pattern
year_table = pd.pivot_table(wind, index='year', values='BE_ON', aggfunc=np.sum)
year_table /= grand_avg

years = np.arange(1980, 2018)
annual_sales_avg = year_table.values.squeeze()
print(len( annual_sales_avg), len(years) )
p1 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 1))
p2 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 2))

plt.figure(figsize=(8,6))
plt.plot(years[:-1], annual_sales_avg, 'ko')
plt.plot(years, p1(years), 'C0-')
plt.plot(years, p2(years), 'C1-')
plt.xlim(1980, 2018.5)
plt.title("Relative wind by Year")
plt.ylabel("Relative wind")
plt.xlabel("Year")
plt.show()

print(f"2016 Relative wind by Degree-1 (Linear) Fit = {p1(2018):.4f}")
print(f"2016 Relative wind by Degree-2 (Quadratic) Fit = {p2(2018):.4f}")

# We pick the quadratic fit
annual_growth = p2