import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

%matplotlib inline

import plotly.graph_objs as go

from wordcloud import WordCloud

import geopandas as gpd
df = pd.read_csv('../input/chennai-zomato-restaurants-data/Zomato Chennai Listing 2020.csv')

df.head()
df.replace(to_replace = ['None','Invalid','Does not offer Delivery','Does not offer Dining','Not enough Delivery Reviews','Not enough Dining Reviews'], value =np.nan,inplace=True)

df.isnull().sum()
df['name of restaurant'] = df['Name of Restaurant'].apply(lambda x: x.lower())

df['Top Dishes'] = df["Top Dishes"].astype(str)

df['Top Dishes'] = df['Top Dishes'].apply(lambda x:x.replace('[','').replace(']','').replace("'",'').replace('  ','').split(','))

df['Cuisine'] = df["Cuisine"].astype(str)

df['Cuisine'] = df['Cuisine'].apply(lambda x:x.replace('[','').replace(']','').replace("'",'').replace('  ','').split(','))

df['Features'] = df['Features'].apply(lambda x:x.replace('[','').replace(']','').replace("'",'').replace('  ','').split(','))

df['Dining Rating Count'] = df['Dining Rating Count'].astype("Float32")

df['Delivery Rating Count'] = df['Delivery Rating Count'].astype("Float32")
def locsplit(x):

    if len(x.split(','))==2:

        return x.split(',')[1].replace(' ','')

    else:

        return x



df['Location_2'] = df['Location'].apply(lambda x: locsplit(x))
print(len(df['Location'].unique()))

print(len(df['Location_2'].unique()))
print(df['Location_2'].unique().tolist())
feat_list = [feat.lower() for feats in df['Features'].tolist() for feat in feats]

print(len(set(feat_list)))

print(list(set(feat_list)))
fig = go.Figure(data=[go.Bar(

                x = df['Location_2'].value_counts()[:20].index.tolist(),

                y = df['Location_2'].value_counts()[:20].values.tolist())])



fig.show()
df['name of restaurant'].value_counts()[:25]
bins_r = [0,2.5,4,5]

group_r = ['bad','good','best']

df['Dining Rating'] = df['Dining Rating'].astype(float)

df['Dine_Verdict'] = pd.cut(df['Dining Rating'],bins_r,labels=group_r)

yv = df['Dine_Verdict'].value_counts().tolist()

colors = ['blue','green','red']

fig = go.Figure(data=[go.Bar(x=group_r,y=yv,marker_color=colors)])

fig.show()
loc_price2 = pd.crosstab(df['Location_2'],df['Dine_Verdict'],margins=True,margins_name='Total') 

loc_price3 = loc_price2.sort_values('Total',ascending=False)[1:26]

loc_price3.drop(columns=['Total'],inplace=True)

loc_price3.div(loc_price3.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True,figsize=(20, 10),color = ['g','y','b','r'])
bins_r = [0,3.5,4,5]

group_r = ['bad','good','best']

df['Delivery Rating'] = df['Delivery Rating'].astype(float)

df['Delivery_Verdict'] = pd.cut(df['Delivery Rating'],bins_r,labels=group_r)

yv = df['Delivery_Verdict'].value_counts().tolist()

colors = ['blue','green','red']

fig = go.Figure(data=[go.Bar(x=group_r,y=yv,marker_color=colors)])

fig.show()
loc_price4 = pd.crosstab(df['Location_2'],df['Delivery_Verdict'],margins=True,margins_name='Total') 

loc_price5 = loc_price4.sort_values('Total',ascending=False)[1:26]

loc_price5.drop(columns=['Total'],inplace=True)

loc_price5.div(loc_price5.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True,figsize=(20, 10),color = ['g','y','b','r'])
bins = [0,500,1000,2500,float("inf")]

groups = ['cheap','moderate','pricey','expensive']

df['Cost'] = pd.cut(df['Price for 2'], bins,labels=groups)

yc = df['Cost'].value_counts().tolist()

colors = ['green','orange','blue','red']

fig = go.Figure(data=[go.Bar(x=groups,y=yc,marker_color=colors)])

fig.show()
loc_price0 = pd.crosstab(df['Location_2'],df['Cost'],margins=True,margins_name='Total') 

loc_price1 = loc_price0.sort_values('Total',ascending=False)[1:26]

loc_price1.drop(columns=['Total'],inplace=True)

loc_price1.div(loc_price1.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True,figsize=(20, 10),color = ['g','y','b','r'])
dishes = ' '.join(dish for dish_list in df['Top Dishes'].tolist() for dish in dish_list if dish != np.nan)

wordcloud = WordCloud(background_color='white',stopwords=['nan']).generate(dishes)

figure(figsize=(20, 10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
cuisines = ' '.join(dish for dish_list in df['Cuisine'].tolist() for dish in dish_list if dish != 'Invalid')

wordcloud = WordCloud(background_color='white').generate(cuisines)

figure(figsize=(20, 10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
def veg_status(feat_list):

    if 'Vegetarian Only' in feat_list:

        return 'Yes'

    elif ' Vegetarian Only' in feat_list:

        return 'Yes'

    else:

        return 'No'
df['Vegetarian Status'] = df['Features'].apply(lambda x: veg_status(x))

df['Vegetarian Status'].value_counts()
fig = go.Figure(data=[go.Bar(

                x = df.loc[df['Vegetarian Status'] == 'Yes']['name of restaurant'].value_counts()[:10].index.tolist(),

                y = df.loc[df['Vegetarian Status'] == 'Yes']['name of restaurant'].value_counts()[:10].values.tolist())])



fig.show()
fig = go.Figure(data=[go.Bar(

                x = df.loc[df['Vegetarian Status'] == 'Yes']['Location'].value_counts()[:10].index.tolist(),

                y = df.loc[df['Vegetarian Status'] == 'Yes']['Location'].value_counts()[:10].values.tolist())])



fig.show()
df.loc[df['Dining Rating Count'].nlargest(10).index][['Name of Restaurant','Location_2','Dining Rating Count','Delivery Rating Count']]
df.loc[df['Delivery Rating Count'].nlargest(10).index][['Name of Restaurant','Location_2','Delivery Rating Count','Dining Rating Count']]