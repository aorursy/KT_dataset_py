import pandas as pd

import os



os.chdir('../input/netflix-shows')



df = pd.read_csv('netflix_titles.csv')

new_df = df[['title','director','cast','listed_in','description']]

new_df.head()
df = pd.read_csv('netflix_titles.csv')

df["date_added"] = pd.to_datetime(df['date_added'])

df['year_added'] = df['date_added'].dt.year

df['month_added'] = df['date_added'].dt.month



df['season_count'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" in x['duration'] else "", axis = 1)

df['duration'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" not in x['duration'] else "", axis = 1)

df.head()
from collections import Counter





titles = [title for title in df['title']]

words_in_titles = [word for title in titles for word in title.split() if len(word) > 3]

counter = Counter(words_in_titles)

most_occur_words_in_titles = counter.most_common(20)

most_occur_words_in_titles
df['type'].value_counts().plot.pie(y='type', title='Ile filmów i seriali', figsize=(9, 9), autopct='%1.01f%%')
from plotly import graph_objects as go





d1 = df[df["type"] == "TV Show"]

d2 = df[df["type"] == "Movie"]



col = "year_added"



vc1 = d1[col].value_counts().reset_index()

vc1 = vc1.rename(columns = {col : "count", "index" : col})

vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))

vc1 = vc1.sort_values(col)



vc2 = d2[col].value_counts().reset_index()

vc2 = vc2.rename(columns = {col : "count", "index" : col})

vc2['percent'] = vc2['count'].apply(lambda x : 100*x/sum(vc2['count']))

vc2 = vc2.sort_values(col)



trace1 = go.Scatter(

                    x=vc1[col], 

                    y=vc1["count"], 

                    name="TV Shows", 

                    marker=dict(color = 'rgb(249, 6, 6)',

                             line=dict(color='rgb(0,0,0)',width=1.5)))



trace2 = go.Scatter(

                    x=vc2[col], 

                    y=vc2["count"], 

                    name="Movies", 

                    marker= dict(color = 'rgb(26, 118, 255)',

                              line=dict(color='rgb(0,0,0)',width=1.5)))



layout = go.Layout(hovermode= 'closest', title = 'Content added over the years' , xaxis = dict(title = 'Year'), yaxis = dict(title = 'Count'),template= "plotly_dark")

fig = go.Figure(data = [trace1, trace2], layout=layout)

fig.show()
df['rating'].value_counts()[:12].plot.pie(figsize=(10, 10))
df1 = df[df["type"] == "TV Show"]

df2 = df[df["type"] == "Movie"]



temp_df1 = df1['rating'].value_counts().reset_index()

temp_df2 = df2['rating'].value_counts().reset_index()



# temp_df1.plot.bar()



# create trace1

trace1 = go.Bar(

                x = temp_df1['index'],

                y = temp_df1['rating'],

                name="TV Shows",

                marker = dict(color = 'rgb(249, 6, 6)',

                             line=dict(color='rgb(0,0,0)',width=1.5)))



# create trace2 

trace2 = go.Bar(

                x = temp_df2['index'],

                y = temp_df2['rating'],

                name = "Movies",

                marker = dict(color = 'rgb(26, 118, 255)',

                              line=dict(color='rgb(0,0,0)',width=1.5)))





layout = go.Layout(template= "plotly_dark",title = 'RATING BY CONTENT TYPE', xaxis = dict(title = 'Rating'), yaxis = dict(title = 'Count'))

fig = go.Figure(data = [trace1, trace2], layout = layout)

fig.show()
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns





df['Genres'] = df['listed_in'].str.extract('([A-Z]\w{2,})', expand=True)

temp_df = df['Genres'].value_counts().reset_index()



labels=np.array(temp_df['index'])

sizes=np.array(temp_df['Genres'])



fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.bar(labels, sizes)

plt.xticks(rotation=-90)



# Jak widać najczęstszym gatunkiem na Netflixie są:

#   1. Dramaty

#   2. Komedie

#   3. M+iędzynarodowe
from sklearn.preprocessing import MultiLabelBinarizer



data= df['listed_in'].astype(str).apply(lambda s : s.replace('&',' ').replace(',', ' ').split()) 



mlb = MultiLabelBinarizer()

res = pd.DataFrame(mlb.fit_transform(data), columns=mlb.classes_)

corr = res.corr()

corr = corr.round(1)



fig, ax = plt.subplots(figsize=(30, 30))

sns.heatmap(corr, annot=True, ax=ax)

plt.show()



# Widzimy, że garunki typu Music - Musical, Children - Family, Stand-up - Comedy, Spirituality - Faith są maksymalnie skorelowane, co jest zgodne z rzeczywistością.
temp_df1 = df['release_year'].value_counts().reset_index()



trace1 = go.Bar(

                x = temp_df1['index'],

                y = temp_df1['release_year'])

layout = go.Layout(template= "plotly_dark",title = 'Wydania w latach' , xaxis = dict(title = 'Rating'), yaxis = dict(title = 'Count'))

fig = go.Figure(data = [trace1], layout = layout)

fig.show()
df1 = df[df["type"] == "TV Show"]

df2 = df[df["type"] == "Movie"]



temp_df1 = df1['release_year'].value_counts().reset_index()

temp_df2 = df2['release_year'].value_counts().reset_index()



trace1 = go.Bar(

                x = temp_df1['index'],

                y = temp_df1['release_year'],

                name="TV Shows")



trace2 = go.Bar(

                x = temp_df2['index'],

                y = temp_df2['release_year'],

                name = "Movies")



layout = go.Layout(template= "plotly_dark", title = 'Zawartosc w latach', xaxis = dict(title = 'Rok'), yaxis = dict(title = 'Liczba'))

fig = go.Figure(data = [trace1, trace2], layout = layout)

fig.show()
trace = go.Histogram(x = df['duration'],

                     xbins=dict(size=0.7))



layout = go.Layout(template= "plotly_dark", title = 'Rozkład długości filmów', xaxis = dict(title = 'Minuty'), yaxis=dict(title = "Ilosc"))

fig = go.Figure(data = [trace], layout = layout)

fig.show()
print("Podaj liczbe panstw uwzglednionych w rankingu:")

country_count = int(input())





temp_df = df['country'].value_counts().reset_index()[:country_count]



trace1 = go.Bar(

                x = temp_df['index'],

                y = temp_df['country'])



layout = go.Layout(template= "plotly_dark",title = f'{country_count} państw z najwieksza produkcja', xaxis = dict(title = 'Panstwo'), yaxis = dict(title = 'Produkcja'))

fig = go.Figure(data = [trace1], layout = layout)

fig.show()
temp_df1 = df[df['type']=='TV Show']

categories1 = ", ".join(temp_df1['director'].fillna("")).split(", ")

counter_list = Counter(categories1).most_common(11)

counter_list = [_ for _ in counter_list1 if _[0] != ""]

labels1 = [el[0] for el in counter_list][::-1]

values1 = [el[1] for el in counter_list][::-1]



trace1 = go.Bar(

                x = labels1,

                y = values1,

                marker = dict(color = 'rgb(255,51,153)',

                              line=dict(color='rgb(0,0,0)',width=1.5))

               )



layout = go.Layout(template= "plotly_dark", title = 'Top 10 rezyserow', xaxis = dict(title = 'Rezyser'), yaxis = dict(title = 'Liczba'))

fig = go.Figure(data = [trace1], layout = layout)

fig.show()
import pandas as pd

import os





# os.chdir('../input/netflix-shows')



df = pd.read_csv('netflix_titles.csv')

new_df = df[['title','director','cast','listed_in','description']]

new_df.head()
df.columns
!pip install rake-nltk

from rake_nltk import Rake

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer
new_df.dropna(inplace=True)



blanks = [] 



col=['title','director','cast','listed_in','description']

for i,col in new_df.iterrows():

    if type(col)==str:         

        if col.isspace():         

            blanks.append(i)     



new_df.drop(blanks, inplace=True)

new_df
new_df['Key_words'] = ""



for index, row in new_df.iterrows():

    description = row['description']

    

    r = Rake()

    r.extract_keywords_from_text(description)

    key_words_dict_scores = r.get_word_degrees()

    row['Key_words'] = list(key_words_dict_scores.keys())



new_df.drop(columns = ['description'], inplace = True)
new_df['cast'] = new_df['cast'].map(lambda x: x.split(',')[:3])

new_df['listed_in'] = new_df['listed_in'].map(lambda x: x.lower().split(','))

new_df['director'] = new_df['director'].map(lambda x: x.split(' '))



for index, row in new_df.iterrows():

    row['cast'] = [x.lower().replace(' ','') for x in row['cast']]

    row['director'] = ''.join(row['director']).lower()
new_df.set_index('title', inplace = True)

new_df.head()
new_df['bag_of_words'] = ''

columns = new_df.columns

for index, row in new_df.iterrows():

    words = ''

    for col in columns:

        if col != 'director':

            words = words + ' '.join(row[col])+ ' '

        else:

            words = words + row[col]+ ' '

    row['bag_of_words'] = words

    

new_df.drop(columns = [col for col in new_df.columns if col!= 'bag_of_words'], inplace = True)
new_df.head()
count = CountVectorizer()

count_matrix = count.fit_transform(new_df['bag_of_words'])



indices = pd.Series(new_df.index)

indices[:5]
cosine_sim = cosine_similarity(count_matrix, count_matrix)

cosine_sim
def recommendations(Title, cosine_sim = cosine_sim):

    recommended_movies = []

    idx = indices[indices == Title].index[0]

    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    top_10_indexes = list(score_series.iloc[1:11].index)

    

    for i in top_10_indexes:

        recommended_movies.append(list(new_df.index)[i])

        

    return recommended_movies
recommendations('The Two Popes')
print(recommendations('Automata'))

recommendations('6 Years')
recommendations('Christine')