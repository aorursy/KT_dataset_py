import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv("../input/FAO.csv", encoding='latin1')

df.head()

year_list = df.iloc[:,df.columns.str.startswith('Y')].columns

year_list

# Extracting the last ten years

year_list = year_list[len(year_list)-10:len(year_list)]

year_list

# Getting a general idea of Food and Feed of all Countries yearwise

df_new = df.pivot_table(values=year_list, columns='Element',index=['Area'], aggfunc='sum')

df_new.head()

# Applying transpose to the above data to get it ready for work

df_foody = df_new.T

df_foody.head()
# Extracting only food produced

df_food = df_foody.xs('Food', level=1, axis=0)

df_food.head()


# Finding the Top 5 food producer countries over the 10 years

df_food_total = df_food.sum(axis=0).sort_values(ascending=False).head()

df_food_total
df_food_total.plot(kind='bar', title='Topmost 5 Food Producer Countries', color='red')
# Visualising the food produced by top 5 countries over the 10 years

plt.figure(figsize = (10,6))

for i in df_food_total.index:

    year = df_food[i]

    plt.plot(year, marker='p')

    plt.xticks(df_food.index, rotation='horizontal')

    plt.legend(loc='best')
# extracting the food consumers

df_feed =  df_new.T.xs('Feed',level=1, axis=0)

df_feed.head()

# Top 5 food consumers

df_feed_total = df_feed.sum(axis=0).sort_values(ascending=False).head()

df_feed_total
df_feed_total.plot(kind='bar', title="Topmost 5 Food Consumer Countries", color='yellow')

plt.figure(figsize = (10,6))

for i in df_feed_total.index:

    year = df_feed[i]

    plt.plot(year, marker='p')

    plt.xticks(df_feed.index, rotation='horizontal')

    plt.legend(loc='best')
for j in df_food_total.index:

    plt.figure(figsize=(6,3))

    plt.plot(df_feed[j], marker='o', color='y')

    plt.plot(df_food[j], marker='o', color='r')

    plt.xticks(df_feed.index, rotation='vertical')

    plt.legend(loc='best')

    plt.show()
# dropping the unnecessary Columns

df.drop(['Area Abbreviation', 'Area Code', 'Item Code', 'Element Code', 'Unit', 'latitude', 'longitude'], axis=1, inplace=True)

df.head()
df_temp = df.set_index(['Element','Area','Item'])
df_temp.head()
food = df_temp.xs('Food', level=0)

df_item = (df.pivot_table(values =year_list, columns='Element',index=['Item'], aggfunc='sum')).T
df_top_food = df_item.xs('Food',level=1).sum(axis=0).sort_values(ascending=False).head(10)
# Top 10 highest produced food

df_top_food

df_top_food.plot(kind='bar', title='Top 10 highest produced Food', color='blue')

feed = df_temp.xs('Feed', level=0)

df_top_feed = df_item.xs('Feed',level=1).sum(axis=0).sort_values(ascending=False).head(10)
df_top_feed



df_top_feed.plot(kind='bar', title='Top 10 highest Food consumed', color='green')
top_food = df_top_food.head(1).index[0]

top_food
# Extracting the Top 10 countries producing 'Milk - Excluding Butter'

top_food_producing_countries = df_temp.xs('Food',level=0).xs(top_food, level=1).sum(axis=1).sort_values(ascending=False).head(10)

top_food_producing_countries
top_food_producing_countries.plot(kind='bar',title=f'Top 10 Countries producing top produced food ({top_food})', color='green')
# top most consumed food

top_feed = df_top_feed.head(1).index[0]

top_feed

# Here are the Top 10 countries producing 'Cereals - Excluding Beer'

top_feed_producing_countries = df_temp.xs('Feed',level=0).xs(top_feed, level=1).sum(axis=1).sort_values(ascending=False).head(10)

top_food_producing_countries


# Here are the Top 10 countries consuming 'Milk - Excluding Butter'

top_food_consuming_countries = df_temp.xs('Feed',level=0).xs(top_food, level=1).sum(axis=1).sort_values(ascending=False).head(10)

top_food_consuming_countries
# Here are the Top 10 Countries consuming 'Cereals - Excluding Beer'

top_feed_consuming_countries = df_temp.xs('Feed',level=0).xs(top_feed, level=1).sum(axis=1).sort_values(ascending=False).head(10)

top_feed_consuming_countries
top_feed_consuming_countries.plot(kind='bar',title=f'Top 10 Countries Consuming the top consumed food -> {top_feed}', color='green')

df_india_food = df_temp.xs('Food',level=0).xs('India',level=0).sum(axis=1).sort_values(ascending=False).drop_duplicates().head(10)

df_india_food

df_india_food.plot(kind='bar', title='Top 10 Food Item produced', color='blue')


df_india_feed = df_temp.xs('Feed',level=0).xs('India',level=0).sum(axis=1).sort_values(ascending=False).drop_duplicates().head(10)

df_india_feed

df_india_feed.plot(kind='bar', title='Top 10 Food Item Consumed',color='red')
df_temp.xs('India', level=1).xs('Food', level=0).sum(axis=0).tail(10).plot(kind='line', color='green')

df_temp.xs('India', level=1).xs('Feed', level=0).sum(axis=0).tail(10).plot(kind='line',title='Food vs Feed in India', color='red')