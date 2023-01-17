# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.cm as cm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
scrubbed = pd.read_csv('../input/scrubbed.csv',low_memory=False)
scrubbed.head()
scrubbed.info()
plt.figure(figsize=(10,5))
sns.heatmap(scrubbed.isnull(),cbar=False,yticklabels='',cmap='viridis')
clean_df = scrubbed.fillna(value=0)
plt.figure(figsize=(10,5))
sns.heatmap(clean_df.isnull(),cbar=False,yticklabels='',cmap='viridis')
clean_df['country'] = clean_df['country'].map({'us': 'USA', 'gb': 'UK', 'ca': 'Canada', 'au': 'Australia', 'de': 'Germany'})
clean_df['country'].unique()
clean_df['datetime'] = pd.to_datetime(clean_df['datetime'],errors='coerce')
shape_unknown = clean_df[(clean_df['shape'] == 'unknown') | (clean_df['shape'] == 'other') | (clean_df['shape'] == '')]
shape_unknown_count = shape_unknown['shape'].count()
print("Shape couldn't be identified for {} sightings.".format(shape_unknown_count))
shape_known = clean_df.drop(clean_df[(clean_df['shape'] == 'unknown') | (clean_df['shape'] == 'other') | (clean_df['shape'] == '')].index)
shape_known_count = shape_known['city'].count()
print("Shape could be identified for {} sightings.".format(shape_known_count))
percent_shape_known = ((shape_known_count)/(shape_known_count + shape_unknown_count))*100
print("{}% of the time, shapes were identified.".format(percent_shape_known.round(2)))
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
ax = sns.barplot(x =shape_known['shape'].value_counts().head().index, y = shape_known['shape'].value_counts().head().values, color = 'darkslategrey')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xlabel('Shape').set_size(20)
ax.set_ylabel('Count').set_size(20)
ax.set_title('TOP 5 SHAPES').set_size(20)
plt.tight_layout()
for p in ax.patches:
        p.set_width(0.5)
        
most_sightings = clean_df.groupby('country')['shape'].size().sort_values(ascending=False)
most_sightings
plt.figure(figsize=(10,10))
most_sightings.head(4).plot(kind='barh',fontsize=20)
top_3_shapes_of_each_country = pd.DataFrame(clean_df.groupby('country')['shape'].value_counts().groupby(level=0).head(3))
top_3_shapes_of_each_country.columns = ['Count']
top_3_shapes_of_each_country
hours_most_sightings = clean_df['datetime'].dt.hour.value_counts()
years_most_sightings = clean_df['datetime'].dt.year.value_counts().head()
month_most_sightings = clean_df['datetime'].dt.month.value_counts()
def top_years(year):
    if year in years_most_sightings.index:
        return year
hour_vs_year = clean_df.pivot_table(columns=clean_df['datetime'].dt.hour,index=clean_df['datetime'].dt.year.apply(top_years),aggfunc='count',values='city')
hour_vs_year.columns = hour_vs_year.columns.astype(int)
hour_vs_year.columns = hour_vs_year.columns.astype(str) + ":00"
hour_vs_year.index = hour_vs_year.index.astype(int)
hour_vs_year
def pie_heatmap(table, cmap='coolwarm_r', vmin=None, vmax=None,inner_r=0.25, pie_args={}):
    n, m = table.shape
    vmin= table.min().min() if vmin is None else vmin
    vmax= table.max().max() if vmax is None else vmax

    centre_circle = plt.Circle((0,0),inner_r,edgecolor='black',facecolor='white',fill=True,linewidth=0.25)
    plt.gcf().gca().add_artist(centre_circle)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
  
    for i, (row_name, row) in enumerate(table.iterrows()):
        labels = None if i > 0 else table.columns
        wedges = plt.pie([1] * m,radius=inner_r+float(n-i)/n, colors=[cmapper.to_rgba(x) for x in row.values], 
            labels=labels, startangle=90, counterclock=False, wedgeprops={'linewidth':-1}, **pie_args)
        plt.setp(wedges[0], edgecolor='grey',linewidth=1.5)
        wedges = plt.pie([1], radius=inner_r+float(n-i-1)/n, colors=['w'], labels=[row_name], startangle=-90, wedgeprops={'linewidth':0})
        plt.setp(wedges[0], edgecolor='grey',linewidth=1.5)



plt.figure(figsize=(8,8))
plt.title("Timewheel of Hour Vs Year",y=1.08,fontsize=30)
pie_heatmap(hour_vs_year, vmin=-20,vmax=80,inner_r=0.2)
plt.figure(figsize=(10,8))
ax = sns.heatmap(hour_vs_year)
ax.set_xlabel('Hours').set_size(20)
ax.set_ylabel('Year').set_size(20)
month_vs_year = clean_df.pivot_table(columns=clean_df['datetime'].dt.month,index=clean_df['datetime'].dt.year.apply(top_years),aggfunc='count',values='city')
month_vs_year.columns = month_vs_year.columns.astype(int)
month_vs_year
plt.figure(figsize=(10,8))
ax = sns.heatmap(month_vs_year,cmap='coolwarm')
ax.set_xlabel('Month').set_size(20)
ax.set_ylabel('Year').set_size(20)
