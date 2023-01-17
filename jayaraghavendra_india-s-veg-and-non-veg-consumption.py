# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.express as px
import seaborn as sns
fp = "/kaggle/input/india-2020-gisdata/Igismap/Indian_States.shp"
map_df = gpd.read_file(fp)
map_df.keys()
df = pd.read_csv("/kaggle/input/foodvnv/food.csv")

df.style.highlight_max(color='Yellow').highlight_min(color='lightblue')

df.sort_values(by = 'States').style.bar(color='lightblue')
df.style.background_gradient(cmap='Blues')
df[['States','non-vegetarians','vegetarians']].groupby(['States']).median().sort_values('non-vegetarians', ascending = True).plot.bar(figsize=(20,9))
df[['States','vegetarians']].groupby(['States']).median().sort_values('vegetarians', ascending = True).plot.barh(figsize=(8,10))
df_nv = df.groupby('States')[['non-vegetarians']].mean().sort_values('non-vegetarians',ascending = False).reset_index()
fig = px.bar(df_nv.head(5), x='States', y='non-vegetarians',
             labels={'non-vegetarians':'NV index'},
             color='non-vegetarians',
             #orientation='h',
             title="Top 5 Non-Vegetarians states")
fig.show()
df_v = df.groupby('States')[['non-vegetarians']].mean().sort_values('non-vegetarians').reset_index()
fig = px.bar(df_v.head(5), x='States', y='non-vegetarians',
             labels={'non-vegetarians':'NV index'},
             color='non-vegetarians',
             #orientation='h',
             title="Top 5 Vegetarians states")
fig.show()
px.scatter(df[['States','non-vegetarians']],
          x='States', y='non-vegetarians',
          size='non-vegetarians', color='non-vegetarians',template="plotly_dark")
map_df.keys()
#merge both dataframe with state index
merged = map_df.set_index('st_nm').join(df.set_index('States'))
merged.head()
#Top 5 Non-Veg consumption States in India

fig = px.bar(data_frame = df.nsmallest(25,"non-vegetarians").tail(5),
             y="States",
             x="non-vegetarians",
             orientation='h',
             color="States",
             text="non-vegetarians",
             color_discrete_sequence=px.colors.qualitative.D3,template="plotly_dark")

fig.update_traces(texttemplate='%{text:.2s}', 
                  textposition='inside', 
                  marker_line_color='rgb(255,255,255)', 
                  marker_line_width=2.5, 
                  opacity=0.7)
fig.update_layout(width=800,
                  yaxis=dict(autorange="reversed"),
                  title="Top 5 Non-Veg consumption States in India")
fig.show()
#Top 5 Veg consumption States in India

fig = px.bar(data_frame = df.nsmallest(25,"vegetarians").tail(),
             y="States",
             x="vegetarians",
             orientation='h',
             color="States",
             text="vegetarians",
             color_discrete_sequence=px.colors.qualitative.D3,template="plotly_dark")

fig.update_traces(texttemplate='%{text:.2s}', 
                  textposition='inside', 
                  marker_line_color='rgb(255,255,255)', 
                  marker_line_width=2.5, 
                  opacity=0.7)
fig.update_layout(width=800,
                  yaxis=dict(autorange="reversed"),
                  title="Top 5 Veg consumption States in India")
fig.show()
#df = merged[merged.isna().any(axis=1)]
#df
#create subplot for number of deaths in India
fig2, axis2 = plt.subplots(1, figsize=(18,10))
axis2.axis('off')
axis2.set_title('Non-Veg Consumptions states', fontdict={'fontsize':'25','fontweight':'5'})
merged.plot(column='non-vegetarians',cmap='Reds',ax=axis2, legend=True)
#YlOrRd viridis
#create subplot for number of deaths in India
fig2, axis2 = plt.subplots(1, figsize=(18,10))
axis2.axis('off')
axis2.set_title('Veg Consumptions states', fontdict={'fontsize':'25','fontweight':'5'})
merged.plot(column='vegetarians',cmap='Greens',ax=axis2, legend=True)
#YlOrRd viridis
heat_map = df.corr()
plt.figure(figsize=(12,10))

ax = sns.heatmap(heat_map, annot=True, cmap='viridis') #notation: "annot" not "annote"
ax
df.keys()
V = df["vegetarians"].sum()
NV = df["non-vegetarians"].sum()

Total = V + NV

print(V)
print(NV)
print(Total)
vegetarians_percentage = (V/Total)*100
non_vegetarians_percentage = (NV/Total)*100

print(vegetarians_percentage)
print(non_vegetarians_percentage)
data = [['Vegetarians', 28.142857142857142], ['Non-Vegetarian', 71.85714285714286]] 
  
# Create the pandas DataFrame 
pie_df = pd.DataFrame(data, columns = ['Type', 'data']) 

pie_df
fig = px.pie(pie_df, values='data',
             names='Type',color='Type',
             color_discrete_map={'Non-Vegetarian':'brown','Vegetarians':'green'})

fig.update_layout(title_text='Veg and Non-Veg Share in India')

fig.show()
import plotly.graph_objects as go

# Pie chart
labels = ['Vegetarians', 'Non-Vegetarian']
values = [28.142857142857142, 71.85714285714286]

# pull is given as a fraction of the pie radius
fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0.1])])
fig.update_layout(title_text='Veg vs Non-Veg Share in India')
fig.show()