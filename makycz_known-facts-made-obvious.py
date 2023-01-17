# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import geopandas as gp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import squarify as sq

# matplotlib notebook
# import mpld3
# mpld3.enable_notebook()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# choropleth map> https://github.com/bendoesdata/make-a-map-geopandas/blob/master/README.mda
# treemap> https://gist.github.com/gVallverdu/0b446d0061a785c808dbe79262a37eea
df = pd.read_csv('../input/countries of the world.csv', decimal=',')
df.head()
#df['Region'].unique()
df.rename(columns={df.columns[8]:'GDP'}, inplace=True)
df.describe()
correlation = df.corr()
plt.figure(figsize=(15, 15))
sns.heatmap(correlation, 
            xticklabels = correlation.columns.values, 
           yticklabels = correlation.columns.values, 
            cmap = "inferno_r");
# map visualization
map_df = gp.read_file(gp.datasets.get_path('naturalearth_lowres'))
map_df.head()
map_df.rename(columns={map_df.columns[2]:'Country'}, inplace=True)

# remove white-space from column Country in the df, rename GDP
df['Country'] = df['Country'].str.strip()
df['Region'] = df['Region'].str.strip()
merged_df = pd.merge(map_df, df, on = 'Country', how = 'inner')
merged_df.head()

# Fill in missing birthrate value in Sub-Saharan Africa with mean value in this region
# merged_df.dtypes
merged_df['Birthrate'].isna().sum()
merged_df['Birthrate'].fillna(merged_df[merged_df['Region'] == 'SUB-SAHARAN AFRICA']['Birthrate'].mean(), inplace = True)
merged_df['Birthrate'].isna().sum()
# Set the variables, range for the choropleth and set the colormap
variable_1 = 'Birthrate'
variable_2 = 'GDP'
vmin_1, vmax_1 = merged_df.Birthrate.min(), merged_df.Birthrate.max()
vmin_2, vmax_2 = merged_df.GDP.min(), merged_df.GDP.max()
cmap = 'hot_r'

# Make a plot
fig, ax = plt.subplots(ncols=2, figsize=(20, 15))
ax = ax.flatten()
merged_df.plot(column=variable_1, cmap=cmap, linewidth=0.9, ax=ax[0], edgecolor='0.6')
merged_df.plot(column=variable_2, cmap=cmap, linewidth=0.9, ax=ax[1], edgecolor='0.6')

# Remove the axis
ax[0].axis('off')
ax[1].axis('off')

# Add a title
ax[0].set_title('Birthrate worldwide', \
              fontdict={'fontsize': '40',
                        'fontweight' : '3'})

ax[1].set_title('GDP worldwide', \
              fontdict={'fontsize': '40',
                        'fontweight' : '3'})

# Create colorbar as a legend
sm_1 = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin_1, vmax=vmax_1))
sm_1._A = []
cbar = fig.colorbar(sm_1, ax=ax[0], shrink = 0.8, orientation = 'horizontal')

sm_2 = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin_2, vmax=vmax_2))
sm_2._A = []
cbar = fig.colorbar(sm_2, ax=ax[1], shrink = 0.8, orientation = 'horizontal')
# Prepare dataset with average GDP in region for the table

avg_GDP = df.groupby(['Region'])['GDP'].mean().sort_values(ascending = False).round(0).to_frame()
avg_GDP.rename(columns={avg_GDP.columns[0]:'Average GDP in region'}, inplace=True)
avg_GDP

# Make plot and table
fig = sns.catplot(x = 'GDP', y = 'Region', data = df, kind = 'box', height = 10)
plt.title('Distribution of GDP in world regions')
plt.table(cellText = avg_GDP.values,
          rowLabels = avg_GDP.index,
          colLabels = avg_GDP.columns,
          cellLoc = 'right', rowLoc = 'center',
          loc = 'right', bbox = [.99,.05,.3,.5]);

print("Country with highest GDP is", df.loc[df['GDP'].idxmax()]['Country'], ", ", df['GDP'].max())
print("Country with lowest GDP is", df.loc[df['GDP'].idxmin()]['Country'], ", ", df['GDP'].min())
# Prepare dataset for treemap
# Filter just Europe countries, omit countries with very small area

df_filtered = df[(df['Region'].str.contains('EUROPE'))| (df['Region'].str.contains('BALTICS'))]
df_filtered.Region.unique()
treemap_df = df_filtered[['Country', 'Population', 'Area (sq. mi.)']]
treemap_df.columns = ['Country', 'Population', 'Area']
treemap_df = treemap_df.query('Area > 5000')
treemap_df = treemap_df.sort_values(by='Area', ascending=False)
treemap_df.head()

population_density = (treemap_df.Population / treemap_df.Area)
treemap_df['pop_density'] = population_density
treemap_df.head()
# Treemap parameters
x = 0.
y = 0.
width = 200.
height = 200.
cmap= plt.cm.Spectral_r

# Color scale on the population
mini, maxi = treemap_df.Population.min(),treemap_df.Population.max()
norm = mpl.colors.Normalize(vmin=mini, vmax=maxi)
colors = [cmap(norm(value)) for value in treemap_df.Population]

# Labels for squares
labels = ['%s\n%d km2\n%d hab' % (label) for label in zip(treemap_df.Country, treemap_df.Area, treemap_df.pop_density)]

# Make plot
fig = plt.figure(figsize=(18, 18))
fig.suptitle('Population density and area of European countries', fontsize=30)
ax = fig.add_subplot(111, aspect='equal')
ax = sq.plot(treemap_df.Population, color=colors, label=labels, ax=ax, alpha=.9)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Square area is proportional to the country area\n', fontsize=20)

# Create colorbar as a legend
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []
cbar = fig.colorbar(sm, shrink = 0.81, orientation = 'vertical')

fig.text(.76, .83, 'Population', fontsize=18)
fig.text(.5, 0.1,
         'Europe: total area %d km2, total population %d habitants' % (treemap_df.Area.sum(), treemap_df.Population.sum()),
         fontsize=20,
         ha='center')
fig.text(.5, 0.16,
         'Countries with area below 5 000 sq km were omitted',
         fontsize=10,
         ha='center')

plt.show()