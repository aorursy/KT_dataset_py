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
df = pd.read_csv('/kaggle/input/indian-food-101/indian_food.csv')
food_df = df.copy()
food_df.head(5)
# We could also get overall info for the dataset
food_df.info()
# Let's import what we'll need for the analysis and visualization
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
%matplotlib inline
# Let's first check Diet
sns.catplot(x = 'diet' , kind = 'count' , data = food_df)
sns.catplot(x = 'diet' , kind = 'count' , data = food_df, hue='region')
food_df['region'] = food_df['region'].replace('-1','Other')
#Lets check the diet plot with region
sns.catplot(x = 'diet' , kind = 'count' , data = food_df, hue='region')
sns.catplot(x = 'diet', col='state' , col_wrap=4, kind = 'count' , data = food_df, height=4, aspect=.8)
food_df['state'] = food_df['state'].replace('-1','Other')
sns.catplot(x = 'diet', col='state' , col_wrap=4, kind = 'count' , data = food_df, height=4, aspect=.8)
sns.catplot(x = 'course', kind = 'count' , data = food_df)
sns.catplot(x = 'course', kind = 'count' , data = food_df, hue='diet')
sns.catplot(x='flavor_profile', kind='count', data=food_df)
sns.catplot(x='flavor_profile', kind='count', data=food_df, hue='course')
pltState = sns.catplot(x = 'course', col='state' , col_wrap=4, kind = 'count' , data = food_df, height=4, aspect=.8)
pltState.set_xticklabels(food_df['course'], rotation=90)
fig = sns.FacetGrid(food_df, hue="course",aspect=4)
fig.map(sns.kdeplot,'prep_time',shade= True)
oldest = food_df['prep_time'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()
fig = sns.FacetGrid(food_df, hue="course",aspect=4)
fig.map(sns.kdeplot,'cook_time',shade= True)
oldest = food_df['cook_time'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()
ingredientswords = []
for i in range(len(food_df)):
    txt =  food_df["ingredients"][i]
    txt =  str(txt.replace(', ', ',').lower())
    #print(txt)
    listTxt = txt.split(',')
    
    for word in listTxt:
        ingredientswords.append(word.capitalize())
        #print(ingredientswords)
words = Counter(ingredientswords)
ING = WordCloud().fit_words(words)

plt.figure(figsize = (12, 12), facecolor = None) 
plt.imshow(ING, interpolation='bilinear')
plt.axis('off')
plt.margins(x=0, y=0)
plt.show()
dessertwords = []
dessert_df = food_df[food_df['course']=='dessert']
for i in range(len(food_df)):
    if food_df['course'][i]=='dessert':
        txt2 =  food_df["ingredients"][i]
        txt2 =  str(txt2.replace(', ', ',').lower())
    
        listTxt2 = txt2.split(',')
    
        for word in listTxt2:
            dessertwords.append(word.capitalize())
            
words = Counter(dessertwords)
ING2 = WordCloud().fit_words(words)

plt.figure(figsize = (12, 12), facecolor = None) 
plt.imshow(ING, interpolation='bilinear')
plt.axis('off')
plt.margins(x=0, y=0)
plt.show()
df_sorted_desc= food_df.sort_values('prep_time',ascending=False)[:10]
#sns.set_theme(style='whitegrid')
ax = sns.catplot(x='name', y='prep_time', kind='bar', data=df_sorted_desc,height=12, aspect=.7)
ax.set_xticklabels(df_sorted_desc['name'], rotation=90, size=14)

ax1 = sns.catplot(x='name', y='cook_time', kind='bar', data=df_sorted_desc,height=12, aspect=.7)
ax1.set_xticklabels(df_sorted_desc['name'], rotation=90, size=14)
#Lets see the ration with cooking time for top 10 prep_time
df_sorted_desc['ratio_cook_prep_time'] = df_sorted_desc['cook_time']/df_sorted_desc['prep_time']
df_sorted_desc
ax1 = sns.catplot(x='name', y='ratio_cook_prep_time', kind='bar', data=df_sorted_desc,height=12, aspect=.7)
ax1.set_xticklabels(df_sorted_desc['name'], rotation=90, size=14)
df_sorted_asc= food_df.sort_values('prep_time',ascending=True)[:10]
ax = sns.catplot(x='name', y='prep_time', kind='bar', data=df_sorted_asc,height=8, aspect=.7)
ax.set_xticklabels(df_sorted_asc['name'], rotation=90, size=14)
ax1 = sns.catplot(x='name', y='cook_time', kind='bar', data=df_sorted_asc,height=8, aspect=.7)
ax1.set_xticklabels(df_sorted_asc['name'], rotation=90, size=14)
#Lets see the ration with cooking time for top 10 prep_time
df_sorted_asc['ratio_cook_prep_time'] = df_sorted_asc['cook_time']/df_sorted_asc['prep_time']

ax2 = sns.catplot(x='name', y='ratio_cook_prep_time', kind='bar', data=df_sorted_asc,height=12, aspect=.7)
ax2.set_xticklabels(df_sorted_asc['name'], rotation=90)
df_sorted_asc
df_updated = food_df[food_df.prep_time != -1]
df_sorted_asc2= df_updated.sort_values('prep_time',ascending=True)[:10]
df_sorted_asc2
ax = sns.catplot(x='name', y='prep_time', kind='bar', data=df_sorted_asc2,height=8, aspect=.7)
ax.set_xticklabels(df_sorted_asc2['name'], rotation=90)
ax1 = sns.catplot(x='name', y='cook_time', kind='bar', data=df_sorted_asc2,height=8, aspect=.7)
ax1.set_xticklabels(df_sorted_asc2['name'], rotation=90)
#Lets see the ration with cooking time for top 10 prep_time
df_sorted_asc2['ratio_cook_prep_time'] = df_sorted_asc2['cook_time']/df_sorted_asc2['prep_time']

ax2 = sns.catplot(x='name', y='ratio_cook_prep_time', kind='bar', data=df_sorted_asc2,height=12, aspect=.7)
ax2.set_xticklabels(df_sorted_asc2['name'], rotation=90, size=14)
