# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import calendar

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
bakery_dataset = pd.read_csv("../input/BreadBasket_DMS.csv")
bakery_dataset.dropna()
bakery_dataset = bakery_dataset[bakery_dataset['Item'] != 'NONE']

bakery_dataset['Date'] = pd.to_datetime(bakery_dataset['Date'])
bakery_dataset['Time'] = pd.to_datetime(bakery_dataset['Time'])
bakery_dataset['Year'] = bakery_dataset['Date'].dt.year
bakery_dataset['Month'] = bakery_dataset['Date'].dt.month
bakery_dataset['Day'] = bakery_dataset['Date'].dt.day
bakery_dataset['Weekday'] = bakery_dataset['Date'].dt.weekday
bakery_dataset['Hour'] = bakery_dataset['Time'].dt.hour
# Any results you write to the current directory are saved as output.
bakery_dataset.head(10)
def map_indexes_and_values(df, col):
    df_col = df[col].value_counts()
    x = df_col.index.tolist()
    y = df_col.values.tolist()
    return x, y

weekmap = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}

# count of the five most popular items
popular_items, popular_items_count = map_indexes_and_values(bakery_dataset, 'Item')
plt.bar(popular_items[:5], popular_items_count[:5])
plt.xlabel('Most popular Items')
plt.ylabel('Number of Transactions')
plt.show()
# top items in 2016
first_year_data = bakery_dataset[bakery_dataset['Year'] == 2016]
x, y = map_indexes_and_values(first_year_data, 'Item')
plt.bar(x[:5], y[:5], color='r', label='2016')
plt.xlabel('Most popular Items')
plt.ylabel('Number of Transactions')
plt.legend()
plt.show()
# top items in 2017
second_year_data = bakery_dataset[bakery_dataset['Year'] == 2017]
x, y = map_indexes_and_values(second_year_data, 'Item')
plt.bar(x[:5], y[:5], color='g', label='2017')
plt.xlabel('Most popular Items')
plt.ylabel('Number of Transactions')
plt.legend()
plt.show()
monday_info = bakery_dataset[bakery_dataset['Weekday'] == 0]
item, item_count = map_indexes_and_values(monday_info, 'Item')
plt.bar(item[:5], item_count[:5], color='b', label='Monday')
plt.xlabel('Popular items on Monday')
plt.ylabel('Number of Transactions')
plt.show()
monday_info = bakery_dataset[bakery_dataset['Weekday'] == 1]
item, item_count = map_indexes_and_values(monday_info, 'Item')
plt.bar(item[:5], item_count[:5], color='b', label='Tuesday')
plt.xlabel('Popular items on Tuesday')
plt.ylabel('Number of Transactions')
plt.show()
monday_info = bakery_dataset[bakery_dataset['Weekday'] == 2]
item, item_count = map_indexes_and_values(monday_info, 'Item')
plt.bar(item[:5], item_count[:5], color='b', label='Wednesday')
plt.xlabel('Popular items on Wednesday')
plt.ylabel('Number of Transactions')
plt.show()
monday_info = bakery_dataset[bakery_dataset['Weekday'] == 3]
item, item_count = map_indexes_and_values(monday_info, 'Item')
plt.bar(item[:5], item_count[:5], color='b', label='Thursday')
plt.xlabel('Popular items on Thursday')
plt.ylabel('Number of Transactions')
plt.show()
monday_info = bakery_dataset[bakery_dataset['Weekday'] == 4]
item, item_count = map_indexes_and_values(monday_info, 'Item')
plt.bar(item[:5], item_count[:5], color='b', label='Friday')
plt.xlabel('Popular items on Friday')
plt.ylabel('Number of Transactions')
plt.show()
monday_info = bakery_dataset[bakery_dataset['Weekday'] == 5]
item, item_count = map_indexes_and_values(monday_info, 'Item')
plt.bar(item[:5], item_count[:5], color='b', label='Saturday')
plt.xlabel('Popular items on Saturday')
plt.ylabel('Number of Transactions')
plt.show()
monday_info = bakery_dataset[bakery_dataset['Weekday'] == 6]
item, item_count = map_indexes_and_values(monday_info, 'Item')
plt.bar(item[:5], item_count[:5], color='b', label='Sunday')
plt.xlabel('Popular items on Sunday')
plt.ylabel('Number of Transactions')
plt.show()
# checking when is the first item most popular weekday wise
first_item = bakery_dataset[bakery_dataset['Item'] == popular_items[0]]
weekday, weekday_count = map_indexes_and_values(first_item, 'Weekday')
x2 = [weekmap[x] for x in weekday]
wkmp = {}
for j,x in enumerate(x2):
    wkmp[x] = weekday_count[j]
order = list(weekmap.values())
ordervals = [wkmp[val] for val in order]
plt.bar(order, ordervals, color='gold')
plt.xlabel('Weekday')
plt.ylabel('Number of Transactions')
plt.title('Popularity of '+popular_items[0]+' by weekday')
plt.show()
# checking when is the second item most popular weekday wise
second_item = bakery_dataset[bakery_dataset['Item'] == popular_items[1]]
weekday, weekday_count = map_indexes_and_values(second_item, 'Weekday')
x2 = [weekmap[x] for x in weekday]
wkmp = {}
for j,x in enumerate(x2):
    wkmp[x] = weekday_count[j]
ordervals = [wkmp[val] for val in order]
plt.bar(order, ordervals, color='gold')
plt.xlabel('Weekday')
plt.ylabel('Number of Transactions')
plt.title('Popularity of '+popular_items[1]+' by weekday')
plt.show()
# checking when is the third item most popular weekday wise
third_item = bakery_dataset[bakery_dataset['Item'] == popular_items[2]]
weekday, weekday_count = map_indexes_and_values(third_item, 'Weekday')
x2 = [weekmap[x] for x in weekday]
wkmp = {}
for j,x in enumerate(x2):
    wkmp[x] = weekday_count[j]
ordervals = [wkmp[val] for val in order]
plt.bar(order, ordervals, color='gold')
plt.xlabel('Weekday')
plt.ylabel('Number of Transactions')
plt.title('Popularity of '+popular_items[2]+' by weekday')
plt.show()
# checking when is the fourth item most popular weekday wise
fourth_item = bakery_dataset[bakery_dataset['Item'] == popular_items[3]]
weekday, weekday_count = map_indexes_and_values(fourth_item, 'Weekday')
x2 = [weekmap[x] for x in weekday]
wkmp = {}
for j,x in enumerate(x2):
    wkmp[x] = weekday_count[j]
ordervals = [wkmp[val] for val in order]
plt.bar(order, ordervals, color='gold')
plt.xlabel('Weekday')
plt.ylabel('Number of Transactions')
plt.title('Popularity of '+popular_items[3]+' by weekday')
plt.show()
# checking when is the fifth item most popular weekday wise
fifth_item = bakery_dataset[bakery_dataset['Item'] == popular_items[4]]
weekday, weekday_count = map_indexes_and_values(fifth_item, 'Weekday')
x2 = [weekmap[x] for x in weekday]
wkmp = {}
for j,x in enumerate(x2):
    wkmp[x] = weekday_count[j]
ordervals = [wkmp[val] for val in order]
plt.bar(order, ordervals, color='gold')
plt.xlabel('Weekday')
plt.ylabel('Number of Transactions')
plt.title('Popularity of '+popular_items[4]+' by weekday')
plt.show()
# checking when is the first item most popular hour wise
first_item = bakery_dataset[bakery_dataset['Item'] == popular_items[0]]
hour, hour_count = map_indexes_and_values(first_item, 'Hour')
plt.bar(hour, hour_count, color='maroon')
plt.xlabel('Hour')
plt.ylabel('Number of Transactions')
plt.title('Popularity of '+popular_items[0]+' by the hour')
plt.xticks(range(6,22))
plt.show()
# checking when is the second item most popular hour wise
second_item = bakery_dataset[bakery_dataset['Item'] == popular_items[1]]
hour, hour_count = map_indexes_and_values(second_item, 'Hour')
plt.bar(hour, hour_count, color='maroon')
plt.xlabel('Hour')
plt.ylabel('Number of Transactions')
plt.title('Popularity of '+popular_items[1]+' by the hour')
plt.xticks(range(0,22))
plt.show()
# checking when is the third item most popular hour wise
third_item = bakery_dataset[bakery_dataset['Item'] == popular_items[2]]
hour, hour_count = map_indexes_and_values(third_item, 'Hour')
plt.bar(hour, hour_count, color='maroon')
plt.xlabel('Hour')
plt.ylabel('Number of Transactions')
plt.title('Popularity of '+popular_items[2]+' by the hour')
plt.xticks(range(6,22))
plt.show()
# checking when is the fourth item most popular hour wise
fourth_item = bakery_dataset[bakery_dataset['Item'] == popular_items[3]]
hour, hour_count = map_indexes_and_values(fourth_item, 'Hour')
plt.bar(hour, hour_count, color='maroon')
plt.xlabel('Hour')
plt.ylabel('Number of Transactions')
plt.title('Popularity of '+popular_items[3]+' by the hour')
plt.xticks(range(6,22))
plt.show()
# checking when is the fifth item most popular hour wise
fifth_item = bakery_dataset[bakery_dataset['Item'] == popular_items[4]]
hour, hour_count = map_indexes_and_values(fifth_item, 'Hour')
plt.bar(hour, hour_count, color='maroon')
plt.xlabel('Hour')
plt.ylabel('Number of Transactions')
plt.title('Popularity of '+popular_items[4]+' by the hour')
plt.xticks(range(6,22))
plt.show()
df = bakery_dataset
cross = df.merge(df, on='Transaction')
final = cross[cross['Item_x']>cross['Item_y']].groupby(['Item_x','Item_y']).size()
ax = final.sort_values(ascending=False)[:5].plot(kind='bar')
ax.set_xlabel('Items bought together')
ax.set_ylabel('Number of common transactions shared by the items')