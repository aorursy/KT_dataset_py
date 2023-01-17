import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd 

import os

#import and split out to US only
df = pd.read_csv("../input/FastFoodRestaurants.csv")

df.head()
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 9

sns.barplot(x=df.name.value_counts().index, y=df.name.value_counts(),
           order = df.name.value_counts().iloc[:10].index)
sns.barplot(x=df.province.value_counts().index, y=df.province.value_counts(),
           order = df.province.value_counts().iloc[:10].index)
ohio_subset = df[df['province'] == 'OH']

sns.barplot(x=ohio_subset.name.value_counts().index, y=ohio_subset.name.value_counts(),
           order = ohio_subset.name.value_counts().iloc[:10].index)
GSC_sub = df[df['name'] == 'Gold Star Chili']
GSC_OH_sub = GSC_sub[GSC_sub['province'] == 'OH']

GSC_count = GSC_sub.name.count()
GSC_OH_count = GSC_OH_sub.name.count()

print(float((GSC_OH_count)/(GSC_count)))

sns.barplot(x=GSC_sub.province.value_counts().index, y=GSC_sub.province.value_counts())
ohio_id_subset = ohio_subset.copy()
ohio_id_subset['gsc_id'] = ohio_id_subset['name'].apply(lambda i: 1 if i == 'Gold Star Chili' else 0)

ohio_id_subset['gsc_id'].unique()
lm = sns.lmplot(x="latitude", y="longitude", data=ohio_id_subset, hue = 'gsc_id',fit_reg = False,
          legend = False, markers=["o", "x"])
axes = lm.axes
axes[0, 0].set_title('Coords of Ohio Fast Food Restaurants - (Gold Star in Orange)')
axes[0,0].set_ylim(-86, -79,)