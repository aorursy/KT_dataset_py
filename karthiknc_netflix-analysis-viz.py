import pandas as pd 

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from wordcloud import WordCloud, STOPWORDS
data=pd.read_csv("../input/netflix-shows/netflix_titles_nov_2019.csv")
data.head()
data.shape
data.isna().sum()
data.info()
data.describe()
k=sns.countplot(data["type"])

for b in k.patches:

    k.annotate(format(b.get_height(),'.0f'),(b.get_x()+b.get_width() / 2.,b.get_height()))
plt.figure(1, figsize=(15, 7))

plt.title("Country with maximum content creation")

k=sns.countplot(x = "country", order=data['country'].value_counts().index[0:15] ,data=data,palette='Accent')

for b in k.patches:

    k.annotate(format(b.get_height(),'.0f'),(b.get_x()+b.get_width() / 2.,b.get_height()))
plt.figure(1, figsize=(15, 7))

plt.title("Frequency")

k=sns.countplot(x = "rating", order=data['rating'].value_counts().index[0:15] ,data=data,palette='Accent')

for b in k.patches:

    k.annotate(format(b.get_height(),'.0f'),(b.get_x()+b.get_width()/2.,b.get_height()))
plt.figure(1, figsize=(15, 7))

plt.title("Frequency")

k=sns.countplot(x = "release_year", order=data['release_year'].value_counts().index[0:15] ,data=data,palette='Accent')

for b in k.patches:

    k.annotate(format(b.get_height(),'.0f'),(b.get_x()+b.get_width()/2.,b.get_height()))
plt.figure(figsize=(20,10))

sns.countplot(x='country',hue='type',data=data[data.country.isin(['United States','India','United Kingdom','Japan','Canada'])])
category = set(data['listed_in'].unique())



category_list = []

for x in category:

    for i in x.split(', '):

        category_list.append(i)

distinct_categories = set(category_list)

category_count_dict = dict.fromkeys(distinct_categories)

for c in distinct_categories:

    category_count_dict[c] = data[data['listed_in'].str.contains(c)]['show_id'].count()

category_count_df = pd.DataFrame.from_dict(category_count_dict, orient='index', columns=['Count'])

category_count_df.head()
category_count_df.sort_values(by='Count', ascending=False).plot.bar(figsize=(15,9), legend=False)

plt.xlabel('Category')

plt.ylabel('No. of Shows')

plt.title('Number of Shows on Netflix by Category')
movie=data.query("type=='Movie'")

movie['mins']=movie['duration'].str.split(' ',expand=True)[0]

movie['mins']=movie['mins'].astype(int)

movie['hours']=movie['mins']/60
top20=movie.sort_values(by='hours',ascending=False).head(20)

plt.figure(figsize=(10,7))

sns.barplot(data=top20,y='title',x='hours',hue='country',dodge=False)

plt.legend(loc='lower right')

plt.title('Top 10 movies by RunTime')

plt.xlabel('Hours')

plt.ylabel('Movie name')

plt.show()
tvs=data.query("type=='TV Show'")

tvs['mins']=tvs['duration'].str.split(' ',expand=True)[0]

tvs['mins']=tvs['mins'].astype(int)

tvs['hours']=tvs['mins']/60
top20tvs=tvs.sort_values(by='hours',ascending=False).head(20)

plt.figure(figsize=(10,7))

sns.barplot(data=top20tvs,y='title',x='hours',hue='country',dodge=False)

plt.legend(loc='lower right')

plt.title('Top 10 TV Shows by RunTime')

plt.xlabel('Hours')

plt.ylabel('TV Show name')

plt.show()
indact=[]

ind=data.query('country=="India"')

for i in ind['cast']:

    indact.append(i)

newls=[]

for i in indact:

    newls.append(str(i).split(',')[0])

inddf=pd.DataFrame(newls,columns=['name'])

ind_df=inddf.drop(inddf.query('name=="nan"').index)

ind_df['name'].value_counts().head(20).plot(kind="bar")
us=data[data['country'].str.contains('India')==False] 

usact=[]

for i in us['cast']:

    usact.append(i)

newls1=[]

for i in usact:

    newls1.append(str(i).split(',')[0])

    

usdf=pd.DataFrame(newls1,columns=['name'])

us_df=usdf.drop(usdf.query('name=="nan"').index)

us_df['name'].value_counts().head(20).plot(kind="bar")
wordcloud = WordCloud(width = 1000, height = 600, max_font_size = 200, max_words = 150,

                      background_color='white').generate(" ".join(data.description))



plt.figure(figsize=[10,10])

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()