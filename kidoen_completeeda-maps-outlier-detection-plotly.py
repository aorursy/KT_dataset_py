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
%matplotlib inline
import seaborn as sns
import plotly.express as px
import folium
import geopandas as gpd
plt.style.use("dark_background")

df = pd.read_csv("/kaggle/input/indian-food-101/indian_food.csv")
df.head()
df.isnull().sum()
df.info()
df = df.dropna()
df.describe()

df['course'].unique()
def plotmapfor(item):
    itemdata = df[df['course']==item]
    itemdf =itemdata.state.value_counts().reset_index()
    itemdf.columns = ['state','count']

    fp = "../input/india-states/Igismap/Indian_States.shp"
    map_df = gpd.read_file(fp)

    merged = map_df.set_index('st_nm').join(itemdf.set_index('state'))

    fig, ax = plt.subplots(1, figsize=(20, 12))
    ax.axis('off')
    ax.set_title(f'State Wise Distribution of indian {item}', fontdict={'fontsize': '16', 'fontweight' : '3'})
    merged.plot(column='count', cmap='viridis', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)

plotmapfor("snack")
plotmapfor("main course")
plotmapfor("starter")
plotmapfor("dessert")
df['flavor_profile'].value_counts()
def plotmapfor(item):
    itemdata = df[df['flavor_profile']==item]
    itemdf =itemdata.state.value_counts().reset_index()
    itemdf.columns = ['state','count']

    fp = "../input/india-states/Igismap/Indian_States.shp"
    map_df = gpd.read_file(fp)

    merged = map_df.set_index('st_nm').join(itemdf.set_index('state'))

    fig, ax = plt.subplots(1, figsize=(20, 12))
    ax.axis('off')
    ax.set_title(f'State Wise Distribution of flavour profile = {item}', fontdict={'fontsize': '16', 'fontweight' : '3'})
    merged.plot(column='count', cmap='viridis', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
plotmapfor("spicy")
plotmapfor("sweet")
df.head()
df['diet'].unique()
ax = df['diet'].value_counts().plot(kind='bar',
                                      figsize=(10, 6),
                                     color=["skyblue", "pink"])

ax.set_xticklabels(['vegetarian', 'non vegetarian'],rotation=0,fontsize=12)
ax.set_ylabel('count', fontsize = 12)
ax.set_title("Distribution of data based on diet",fontsize=18)
plt.show()
ax = df['course'].value_counts().plot(kind='bar',
                                      figsize=(10, 6),
                                     color=["royalblue","orange","seagreen","maroon"])

ax.set_xticklabels(['dessert', 'main course', 'starter', 'snack'],rotation=0,fontsize=12)
ax.set_ylabel('count', fontsize = 12)
ax.set_title("Distribution of data based on diet",fontsize=18)
plt.show()
df['flavor_profile'].unique()
ax = df['flavor_profile'].value_counts().plot(kind='bar',
                                      figsize=(10, 6),
                                     color=["royalblue","orange","seagreen","maroon","pink"])

ax.set_xticklabels(['sweet', 'spicy', 'bitter', 'NA', 'sour'],rotation=0,fontsize=12)
ax.set_ylabel('count', fontsize = 12)
ax.set_title("Distribution of data based on flavour profile",fontsize=18)
plt.show()
plt.figure(figsize=(14,8))
sns.distplot(df['prep_time'],color='red')
plt.title("Histogram for prep time",fontsize=24)
plt.show()
plt.figure(figsize=(14,8))
sns.boxplot(df['prep_time'],color='red',orient='v')
plt.title("boxplot for prep time",fontsize=24)
plt.show()
plt.figure(figsize=(14,8))
sns.distplot(df['cook_time'],color='seagreen')
plt.title("Histogram for cook time",fontsize=24)
plt.show()
plt.figure(figsize=(14,8))
sns.boxplot(df['cook_time'],color='seagreen',orient='v')
plt.title("boxplot for cook time",fontsize=24)
plt.show()
df['prep_time'].describe()
# as our data is skewed so we will compute the Interquantile range to calculate the boundaries 
IQR=df['cook_time'].quantile(0.75)-df['cook_time'].quantile(0.25)
lower_bridge=df['cook_time'].quantile(0.25)-(IQR*1.5)
upper_bridge=df['cook_time'].quantile(0.75)+(IQR*1.5)
print(lower_bridge), print(upper_bridge)
#### Extreme outliers
lower_bridge=df['cook_time'].quantile(0.25)-(IQR*3)
upper_bridge=df['cook_time'].quantile(0.75)+(IQR*3)
print(lower_bridge), print(upper_bridge)
data = df.copy()
data.loc[data['cook_time']>=100,'cook_time']=100
plt.figure(figsize=(14,8))
sns.distplot(data['cook_time'],color='red')
plt.title("Histogram for cook time after handling outliers",fontsize=24)
plt.show()
# Handling outliers for prep_time 

# as our data is skewed so we will compute the Interquantile range to calculate the boundaries 
IQR=df['prep_time'].quantile(0.75)-df['prep_time'].quantile(0.25)
lower_bridge=df['prep_time'].quantile(0.25)-(IQR*1.5)
upper_bridge=df['prep_time'].quantile(0.75)+(IQR*1.5)
print(lower_bridge), print(upper_bridge)

#### Extreme outliers
lower_bridge=df['prep_time'].quantile(0.25)-(IQR*3)
upper_bridge=df['prep_time'].quantile(0.75)+(IQR*3)
print(lower_bridge), print(upper_bridge)


data.loc[data['prep_time']>=50,'prep_time']=50
plt.figure(figsize=(14,8))
sns.distplot(data['prep_time'],color='seagreen')
plt.title("Histogram for prep time after handling outliers",fontsize=24)
plt.show()
data['totaltime'] = data['prep_time']+data['cook_time']
data.head()
from wordcloud import WordCloud, STOPWORDS

comment_words = '' 
stopwords = set(STOPWORDS)
for val in data.name: 
    val = str(val) 
    tokens = val.split() 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower()       
    comment_words += " ".join(tokens)+" "
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
                        
plt.figure(figsize = (10,10), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title("Wordcloud for Dishes",fontsize=24)  
plt.show()
from wordcloud import WordCloud, STOPWORDS

comment_words = '' 
stopwords = set(STOPWORDS)
for val in data.ingredients: 
    val = str(val) 
    tokens = val.split() 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower()       
    comment_words += " ".join(tokens)+" "
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
                        
plt.figure(figsize = (10,10), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title("Wordcloud for Ingredients",fontsize=24) 
plt.show()
data.head()
group = data.groupby(['course'])[['prep_time']].mean().reset_index()
fig = px.pie(group,values='prep_time',names='course',title="average preparation time taken for items belonging to each course")
fig.update_traces(rotation=90, pull=0.02, textinfo="percent+label")
fig.show()
group = data.groupby(['course'])[['cook_time']].mean().reset_index()
fig = px.pie(group,values='cook_time',names='course',title="average cooking time taken for items belonging to each course")
fig.update_traces(rotation=90, pull=0.02, textinfo="percent+label")
fig.show()
df = px.data.tips()
fig = px.sunburst(data, path=['region','course','flavor_profile'], values='totaltime',color='region')
fig.show()
df = px.data.tips()
fig = px.sunburst(data, path=['region','course','diet'], values='totaltime',color='region')
fig.show()
def groupbyplot(feature):
    group = data.groupby("name")[feature].mean().sort_values(ascending=False).reset_index().head(10)
    fig = plt.figure(figsize=(16,8))
    fig = px.bar(group, x='name',y=feature,color=feature,title=f"Top 10 dishes based on {feature} taken")
    fig.show()
groupbyplot("totaltime")


