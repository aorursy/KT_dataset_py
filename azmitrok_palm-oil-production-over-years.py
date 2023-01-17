import pandas as pd

import numpy as np



df = pd.read_csv("../input/FAO.csv",  encoding = "ISO-8859-1")

pd.options.mode.chained_assignment = None  # default='warn'



df = df[df > 0]

df = df.dropna(how='any')



df_palm = df[df.Item.str.contains('Palm') & df.Item.str.contains('Oil')]

df_palm.Item.value_counts()
df_selected_food = df_palm

df_selected_food.loc[:,'Percentage'] = df_selected_food.Y2013/df_selected_food.Y2013.sum()

df_selected_food = df_selected_food.sort_values(by='Y2013', ascending=False)

df_selected_food.loc[:,'Cumsum'] = df_selected_food.Percentage.cumsum()



df_selected_food[['Area', 'Y2013', 'Percentage', 'Cumsum']]
df_result = df_selected_food.groupby('Area').sum().sort_values(by='Y2013', ascending=False)



df_result = df_result.drop(['Area Code', 'Item Code', 'Element Code', 'latitude', 'longitude'], axis=1)



import matplotlib.pyplot as plt



df_result = df_result.transpose()



df_result.plot(kind='barh', stacked=True, figsize=(12, 10))

plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))