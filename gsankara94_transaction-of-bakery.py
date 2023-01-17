import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from datetime import datetime 
file = '../input/BreadBasket_DMS.csv'
df = pd.read_csv(file)
df.head()
df.tail()
df.shape
df.info()
# Let's study the unique items first (top 10)
plt.figure(figsize=(10,10))
item_count = df['Item'].value_counts()
item_count[:10].plot(kind='bar')
plt.show()
# Pi-chart 
labels = item_count[:10].index.tolist()
fig = plt.figure(figsize=[10, 10])
ax = fig.add_subplot(111)
ax.pie(item_count[:10],labels=labels,autopct='%1.1f%%')
plt.axis('equal')
plt.show()
datestamp = [datetime.strptime(x, '%Y-%m-%d').date() for x in df['Date']]
timestamp = [datetime.strptime(x,  '%H:%M:%S').time() for x in df['Time']]
def day(hour):
    if hour >= 6 and hour < 12:
        return 'Morning'
    elif hour >= 12 and hour < 15:
        return 'Afternoon'
    else: 
        return 'Evening'
# Extract hour 
time_of_day = [day(x.hour) for x in timestamp]
df['Time of Day'] = time_of_day
df.head()
# Let's group items by time of day and count 
count_by_day = df.groupby(['Time of Day','Item'])['Item'].agg('count')
# Print top 3 for Evening
evening = count_by_day.loc['Evening'].sort_values(ascending=False)[:5]
morning = count_by_day.loc['Morning'].sort_values(ascending=False)[:5]
f, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,6))
evening.plot(kind='bar',ax=ax1)
morning.plot(kind='bar',ax=ax2)
plt.show()
# First let's extract the month numbers 
month_num = [date.month for date in datestamp]
def month_category(month):
    if month >=1 and month <=3:
        return 'Winter'
    elif month >=4 and month <=6:
        return 'Spring'
    elif month >=7 and month<=9:
        return 'Summer'
    else:
        return 'Fall'
seasons = [month_category(month) for month in month_num]
df['Seasons'] = seasons
df.tail()
popular_season = df.groupby(['Seasons','Item'])['Item'].agg('count')
# Winter
Fall = popular_season.loc['Fall'].sort_values(ascending=False)[:5]
Spring = popular_season.loc['Spring'].sort_values(ascending=False)[:5]
Winter=popular_season.loc['Winter'].sort_values(ascending=False)[:5]
#print(popular_season.loc['Summer'].sort_values(ascending=False)[:5])
f, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(14,6))
Fall.plot(kind='bar',ax=ax1)
Spring.plot(kind='bar',ax=ax2)
Winter.plot(kind='bar',ax=ax3)
plt.show()
def coffee_ext(group):
    match = group['Item'].str.contains('Coffee')
    return df.loc[match]

# Let's get the transaction numbers of all the transactions that have coffee.
coffee = df[df['Item'].str.contains('Coffee')]['Transaction'].unique()
# Now that we have all the coffee transactions, we can do a left join with coffee
coffee = pd.DataFrame(coffee,columns=['Transaction'])
coffee_m=coffee.merge(df, left_on='Transaction',right_on='Transaction',how='right')
# Remove all the coffee rows, groupby transaction and tally up the items
coffee_m = coffee_m[~coffee_m.Item.str.contains('Coffee')]['Item'].value_counts()
plt.figure(figsize=(10,6))
coffee_m[:5].plot(kind='bar')
plt.show()