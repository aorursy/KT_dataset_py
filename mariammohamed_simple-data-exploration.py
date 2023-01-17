import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
from plotly.graph_objs import *
%matplotlib inline
data = pd.read_csv('../input/FAO.csv', encoding='latin-1')
pd.set_option("max_column", 100)
data.shape
data.head()
len(data['Area'].unique())
data.Element.unique()
len(data['Element Code'].unique())
len(data.Item.unique())
len(data['Item Code'].unique())
data['Sum Years'] = 0
for year in range(1962, 2014):
    col = 'Y' + str(year)
    data['Sum Years'] = data['Sum Years'] + data[col]
el_size = data.groupby('Element').agg('size')
el_size.values
sns.barplot(el_size.index, el_size.values)
plt.show()
item_area = []
for item, group in data.groupby(['Item', 'Area']):
    item_area.append((item[0], item[1], group.Element.values.tolist()))
only_food = set()
only_feed = set()
food_and_feed = set()
list(map(lambda x: only_feed.add(x[0]), list(filter(lambda x: 'Food' not in x[2], item_area))));
list(map(lambda x: only_food.add(x[0]), list(filter(lambda x: 'Feed' not in x[2], item_area))));
list(map(lambda x: food_and_feed.add(x[0]), list(filter(lambda x: 'Feed' in x[2] and 'Food' in x[2], item_area))));
only_food.intersection(food_and_feed)
only_feed.intersection(food_and_feed)
only_feed.difference(food_and_feed)
only_food.difference(food_and_feed)

data_item_grouped = data.groupby('Item')
max_sum_items = data_item_grouped.agg('max')['Sum Years']
max_sum_items_area = {}
for item, group in data_item_grouped:
#     print(group[group['Sum Years'] == max_sum_items[item]]['Area'].values[0])
#     print(max_sum_items[item])
    max_sum_items_area[item] = group[group['Sum Years'] == max_sum_items[item]]['Area'].values[0]
max_sum_items = max_sum_items.to_dict()
max_sum_items_sorted = sorted(max_sum_items.items(), key=lambda x: x[1], reverse=True)
titles_areas = []
for k, v in max_sum_items_sorted:
    titles_areas.append(max_sum_items_area[k])
items = list(map(lambda x: x[0], max_sum_items_sorted))
values = list(map(lambda x: x[1], max_sum_items_sorted))
titles_areas_items = list(map(lambda x: "(" + x[0] + ")  ,  " + x[1], list(zip(titles_areas, items))))
fig, ax1 = plt.subplots()
sns.barplot(values[:20], items[:20], ax=ax1)
ax1.tick_params(labeltop=False, labelright=True)
ax_2 = ax1.twinx()
ax_2.set_yticks(list(range(20)))
ax_2.set_yticklabels(titles_areas[:20][::-1])
plt.show()
area_2013 = data.groupby('Area')['Y2013'].agg('sum')
area_1961 = data.groupby('Area')['Y1961'].agg('sum')
iplot([go.Choropleth(
    locationmode='country names',
    locations=area_1961.index,
    text=area_1961.index,
    z=area_1961.values
)],filename='1961')
iplot([go.Choropleth(
    locationmode='country names',
    locations=area_2013.index,
    text=area_2013.index,
    z=area_2013.values
)],filename='2013')