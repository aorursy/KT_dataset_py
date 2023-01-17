import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt

import seaborn as sns

%matplotlib inline

df = pd.read_csv('/kaggle/input/zomato_restaurants_in_India.csv',engine="python")

# drop duplicate rows using res_id as it is unique for each restaurant

df.drop_duplicates(subset='res_id',inplace=True)
# Quick info on each columns

df.head()



print("Dataframe shape")

print(df.shape)



df.isnull().sum()



df.city.nunique()

# there are 99 cities



df.city_id.nunique()

# with 83 city IDs



print(df.delivery.unique())



print(df.takeaway.unique())



print(df.opentable_support.unique())
city_df = df.loc[:,['res_id','city']]

res_by_city = city_df.groupby('city').count()

res_by_city.sort_values('res_id',ascending=False,inplace=True)

fig,axes =  plt.pyplot.subplots()

fig.set_size_inches(25,50)

sns.set_style('ticks')

plt.rcParams.update({'font.size': 20,'axes.labelsize':42})

res_by_city_plot = sns.barplot(x='res_id',y=res_by_city.index,data=res_by_city,ax=axes)
res_by_city.head()
res_by_ratings = df.loc[:,['res_id','aggregate_rating']]

res_by_ratings = res_by_ratings.groupby('aggregate_rating').count()

res_by_ratings.sort_values('res_id',ascending=False,inplace=True)

res_by_ratings.head()

fig,axes =  plt.pyplot.subplots()

fig.set_size_inches(30,10)

sns.set_style('ticks')

plt.rcParams.update({'font.size': 28,'axes.labelsize':42})

sns.barplot(x=res_by_ratings.index,y=res_by_ratings.res_id,data=res_by_ratings,ax=axes)
price_category = df.loc[:,['res_id','price_range','average_cost_for_two']]

fig,axes =  plt.pyplot.subplots()

fig.set_size_inches(10,10)

plt.rcParams.update({'font.size': 14,'axes.labelsize':14})

axes.yaxis.set_ticks(np.arange(0, np.max(price_category["average_cost_for_two"])+1, 1000))

axes.scatter(x='price_range',y='average_cost_for_two',data=price_category)
price_category = price_category[price_category['average_cost_for_two']<4000]

fig,axes =  plt.pyplot.subplots()

fig.set_size_inches(10,10)

#axes.yaxis.set_ticks(np.arange(0, np.max(price_category["average_cost_for_two"])+1, 1000))

axes.scatter(x='price_range',y='average_cost_for_two',data=price_category)
cost_vs_ratings = df.loc[:,['rating_text','price_range','city']]

cost_vs_ratings.groupby('rating_text').count()

cost_vs_ratings = cost_vs_ratings[cost_vs_ratings['rating_text'].isin(['Average','Excellent','Good','Poor','Very Good'])]
fig,axes = plt.pyplot.subplots()

fig.set_size_inches(20,10)

#sns.set_style('ticks')

sns.violinplot(x='rating_text',y='price_range',data=cost_vs_ratings)
cost_vs_ratings = df.loc[:,['rating_text','price_range','res_id']]

cost_vs_ratings['rating_range'] = cost_vs_ratings['rating_text'].map({'Poor':1,'Average':2,'Good':3,'Very Good':4,'Excellent':5})

cost_vs_ratings.drop('res_id',axis=1,inplace=True)

sns.heatmap(cost_vs_ratings.pivot_table(index=['price_range'],columns="rating_range",aggfunc="count"),cmap="coolwarm")
df.price_range.value_counts(dropna=True)

res_by_cost_for_two = df.loc[:,['res_id','price_range']]

res_by_cost_for_two = res_by_cost_for_two.groupby('price_range',as_index=False).count()

res_by_cost_for_two.sort_values('res_id',ascending=False,inplace=True)

res_by_cost_for_two
fig,axes =  plt.pyplot.subplots()

fig.set_size_inches(10,10)

sns.set_style('ticks')

plt.rcParams.update({'font.size': 20,'axes.labelsize':20})

sns.barplot(x=res_by_cost_for_two.price_range,y=res_by_cost_for_two.res_id,data=res_by_cost_for_two,ax=axes)
costliest_res = df[df['average_cost_for_two']>4000].sort_values('average_cost_for_two',ascending=False)

costliest_res.shape
fig,axes =  plt.pyplot.subplots()

fig.set_size_inches(10,10)

sns.set_style('ticks')

plt.rcParams.update({'font.size': 20,'axes.labelsize':20})

sns.barplot(x="average_cost_for_two",y="city",data=costliest_res,estimator=sum,ax=axes)
fig,axes = plt.pyplot.subplots()

fig.set_size_inches(20,10)

plt.pyplot.xticks(rotation=90)

sns.swarmplot(y="average_cost_for_two",x="city",data=costliest_res)
ratings_by_city = df.loc[:,['city','aggregate_rating']].groupby(by=['city']).agg(['mean','count'])

ratings_by_city = ratings_by_city.stack(level=0)

ratings_by_city.reset_index(inplace=True)

ratings_by_city.sort_values('count',ascending=False,inplace=True)
fig,axes = plt.pyplot.subplots()

fig.set_size_inches(20,30)

sns.set_style('ticks')

plt.rcParams.update({'font.size': 18,'axes.labelsize':20})

sns.barplot(y="city",x="mean",data=ratings_by_city,ax=axes)
cuisines = set()

def cuisinesSeperator(data):

    for i in data.split(','):

        cuisines.add(i.strip())

    

df.cuisines.dropna().apply(cuisinesSeperator)

print(len(cuisines))
cuisineCount = {}

def cuisineCountMap(data):

    for i in data.split(','):

        if(i.strip() in cuisineCount):

            cuisineCount[i.strip()] = cuisineCount[i.strip()] +1

        else:

            cuisineCount[i.strip()] = 1 

df.cuisines.dropna().apply(cuisineCountMap)

cuisineCount
cuisinesCountDf = pd.DataFrame(list(cuisineCount.items()),columns=['cuisine','count'])

cuisinesCountDf.sort_values(by="count",ascending=False,inplace=True)

fig,axes = plt.pyplot.subplots()

fig.set_size_inches(20,40)

plt.pyplot.xticks(rotation=90)

plt.rcParams.update({'font.size': 20,'axes.labelsize':20})

sns.barplot(y="cuisine",x="count",data=cuisinesCountDf,ax=axes)
from wordcloud import WordCloud
wordcloud = WordCloud(

        background_color='white',

        max_words=100,

        max_font_size=40, 

        scale=3,

        random_state=1 

    ).generate(str(cuisinesCountDf))
fig,axes = plt.pyplot.subplots()

fig.set_size_inches(10,20)

axes.axis('off')

axes.imshow(wordcloud)
cuisineByCity = {}

def cuisine_by_city(data):

    city = data[0]

    cuisine = data[1]

    if city in cuisineByCity:

        for i in cuisine.split(','):

            if(i.strip() in cuisineByCity[city]):

                cuisineByCity[city][i.strip()] = cuisineByCity[city][i.strip()] +1

            else:

                cuisineByCity[city][i.strip()] = 1 

    else:

        cuisineByCity[city] = {}

        for i in cuisine.split(','):

            cuisineByCity[city][i.strip()] = 1

df[['city','cuisines']].dropna().apply(cuisine_by_city, axis=1)
cuisineByCity_df = []

for keys in cuisineByCity.keys():

    for k,v in cuisineByCity[keys].items():

        cuisineByCity_df.append((keys,k,v))

cuisineByCity_df = pd.DataFrame(cuisineByCity_df,columns=['city','cuisine','count'])
cuisineByCity_df.sort_values(by=['city','count'],ascending=False,inplace=True)

cuisineVsCity = cuisineByCity_df.groupby('city',as_index=False).first()
wordcloud = WordCloud(

        background_color='white',

        max_words=10,

        max_font_size=40, 

        scale=3,

        random_state=1 

    ).generate(str(cuisineVsCity['cuisine'].values))

fig,axes = plt.pyplot.subplots()

#fig.set_size_inches(10,20)

axes.axis('off')

axes.imshow(wordcloud)
cuisineVsCity.groupby('cuisine').count()
cityVsEstablishment = df.loc[:,['city','establishment','res_id']].groupby(['city','establishment'],as_index=False).count()

cityVsEstablishment.sort_values(['city','res_id'],ascending=False,inplace=True)

cityVsEstablishment = cityVsEstablishment.groupby('city').first()

wordcloud = WordCloud(

        background_color='white',

        max_words=10,

        max_font_size=40, 

        scale=3,

        random_state=1 

    ).generate(str(cityVsEstablishment['establishment'].values))

fig,axes = plt.pyplot.subplots()

fig.set_size_inches(10,20)

axes.axis('off')

axes.imshow(wordcloud)
cityVsEstablishment['establishment'].unique()