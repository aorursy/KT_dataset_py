import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns



import cufflinks as cf

#import chart_studio.plotly as py

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots



from collections import Counter



cf.go_offline()



from wordcloud import WordCloud, ImageColorGenerator



%matplotlib inline
# df = pd.read_csv("MoviesOnStreamingPlatforms_updated.csv")

# df_countries = pd.read_csv("FilmsByCountry.csv")



# Kaggle



df = pd.read_csv("../input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv")

df_countries = pd.read_csv("../input/filmsbycountry/FilmsByCountry.csv")
df.head()
df.info()
# Let´s see what data we are missing



sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap="viridis")
netflix_movies = len(df[df["Netflix"]==1].index)

hulu_movies = len(df[df["Hulu"]==1].index)

prime_movies = len(df[df["Prime Video"]==1].index)

disney_movies = len(df[df["Disney+"]==1].index)



print("Number of movies on each platform:")

print("\n")

print(f"Netflix:\t {netflix_movies}")

print(f"Hulu:\t\t {hulu_movies}")

print(f"Prime Video:\t {prime_movies}")

print(f"Disney+:\t {disney_movies}")
values = [netflix_movies,hulu_movies,prime_movies,disney_movies]

labels = ["Netflix","Hulu","Prime Video","Disney+"]

explode = (0.1, 0, 0, 0)



fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])

fig.show()
df_countries.head()
data = df_countries



n = 30



fig = px.bar(df_countries.sort_values(by="Number of Films",ascending=False).reset_index().iloc[:n],

             x='Country', y='Number of Films',

             color='Number of Films',

             title=f"Top {n} countries with most filmed films",

             height=400)

fig.show()
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
fig = px.choropleth(df_countries, locations="CODE",

                    color="Number of Films",

                    hover_name="Country",

                    color_continuous_scale=px.colors.sequential.Plasma,

                    title="Number of films filmed per country",

                    height=500,

                    width=800)

fig.show() 
df.drop("Rotten Tomatoes",axis=1,inplace=True)
n = 30



top_ratings = df.sort_values(by="IMDb",ascending=False).reset_index().iloc[:n]



fig = px.bar(top_ratings,

             x = "Title", y='IMDb',

             color='IMDb',

             hover_data=["Directors","IMDb"],

             title=f"Top {n} best rated movies",

             height=1000,

             color_continuous_scale=px.colors.sequential.Viridis)

fig.show()
df_year = df.groupby("Year").mean()

df_year.drop(["Unnamed: 0","ID",'Netflix', 'Hulu', 'Prime Video', 'Disney+', 'Type'],inplace=True,axis=1)

df_year["Movie Count"] = df.groupby("Year").count()["ID"]

df_year.head()
# Add a new column to our dataframe that indicates how many movies were made that year



# Get a dictionary from the df_year dataframe, where keys=year and values=movies made



d = df_year["Movie Count"].to_dict()

d.pop('IMDb', None)

d.pop("Runtime",None)



# Create our new column using the map function



df['Movie Count'] = df['Year'].map(d)

df.head()
x = df["Year"]

y = df["IMDb"]



plt.figure(figsize=(8,8))

sns.scatterplot(x,y,data=df,hue="Movie Count")
Rating = df_year["IMDb"]

Year = df_year.index



fig = px.line(df_year, x=Year, y=Rating, 

              line_shape="spline", render_mode="svg",

              hover_name=df.groupby("Year").mean().index,

              title="Average Movie Ratings 1902-2020")

fig.show()
fig = px.box(df, x="Year", y="IMDb",color="Year")

fig.show()
fig = px.scatter(df,x="Runtime", y="IMDb",color="Year",

                 marginal_x="histogram",

                 marginal_y="histogram",

                 hover_name="Title",hover_data=["Age"])

fig.show()
print(df.groupby("Age").count()["ID"])



sns.countplot(data=df,x="Age")
height = 800



fig2 = px.scatter(df.dropna(),x="Runtime", y="IMDb",color="Age",

                marginal_x="histogram",

                hover_name="Title",hover_data=["Year"],

                height=height)



fig1 = px.scatter(df.dropna(),x="Year", y="IMDb",color="Age",

                marginal_x="histogram",

                hover_name="Title",hover_data=["Runtime"],

                height=height)

fig1.show()

fig2.show()
data = df



n = 30



fig = px.bar(df.sort_values(by="Runtime",ascending=False).reset_index().iloc[:n],

             x = "Title", y='Runtime',

             color='Runtime',

             hover_data=["Directors","Year","IMDb"],

             title=f"Top {n} longest movies",

             height=800)

fig.show()
fig = px.bar(df.sort_values(by="Runtime",ascending=False).reset_index().iloc[2:n],

             x = "Title", y='Runtime',

             color='Runtime',

             hover_data=["Directors","Year","IMDb"],

             title=f"Top {n} longest movies",

             height=800)

fig.show()
plt.figure(figsize=(10,16))

sns.barplot(x="Runtime",y="Title",data=df.sort_values(by="Runtime",ascending=False).reset_index().iloc[2:102])
Runtime = df_year["Runtime"]

Year = df_year.index



#LIneplot



fig = px.line(df_year, x=Year, y=Runtime, 

              line_shape="spline", render_mode="svg",

              hover_name=df.groupby("Year").mean().index,

              title="Average Movie Runtimes 1902-2020")

fig.show()



print("\n")

print("As we did before, we will plot a boxplot to clear thigs up ever more")



# Boxplot



fig = px.box(df.sort_values(by="Runtime",ascending=False).iloc[2:], x="Year", y="Runtime",

             color="Year",height=1000,

             hover_data=["Title"])

fig.show()
# Create a new dataframe specific for directors and the number of movies they directed



directors = pd.DataFrame(df.groupby(["Directors"]).count()["ID"])

directors["No. of Films"] = directors["ID"]

directors.drop(["ID"],axis=1,inplace=True)
data = directors.sort_values(by="No. of Films",ascending=False).reset_index().iloc[:100]



plt.figure(figsize=(10,16))

sns.barplot(x="No. of Films",y="Directors",data=data)
n = 40

data = directors.sort_values(by="No. of Films",ascending=False).reset_index().iloc[:n]



fig = px.bar(data,

             x = "Directors", y='No. of Films',

             color='No. of Films',

             title=f"Top {n} directors with most directed movies")

fig.show()
# Create directors dict where key=name and value=number of directors



directors = {}



for i in df["Directors"].dropna():

    #print(i,len(i.split(",")))

    directors[i] = len(i.split(","))

    

# Add this information to our dataframe as a new column



df["Number of Directors"] = df['Directors'].map(directors)



# Sort by number of directors and show head



df.sort_values(by="Number of Directors",ascending=False).reset_index().head()
data = df.sort_values(by="Number of Directors",ascending=False).reset_index().iloc[:30]



plt.figure(figsize=(10,6))

sns.barplot(x="Number of Directors",y="Title",data=data)
# Directors who directed the best and worst IMDb ranked movies



n = 30

x="Directors"



data1 = df.groupby(by="Directors").mean().sort_values(by="IMDb",ascending=False).reset_index().iloc[:n]

data2 = df.groupby(by="Directors").mean().sort_values(by="IMDb",ascending=True).reset_index().iloc[2:n]



# For the worst IMDb average we droped the worst two because they had an average of 0 and it´s not considered representative



fig = px.bar(data1,x=data1["Directors"],y=data1["IMDb"],color="IMDb",

             title=f"Top {n} Directors with the highest averaged movie ratings")

fig.show()



fig = px.bar(data2,x=data2["Directors"],y=data2["IMDb"],color="IMDb",

            title=f"Top {n} Directors with the lowest averaged movie ratings")

fig.show()
# Directors who directed the longest and shortest movies



n = 30

x="Directors"



data1 = df.groupby(by="Directors").mean().sort_values(by="Runtime",ascending=False).reset_index().iloc[:n]

data2 = df.groupby(by="Directors").mean().sort_values(by="Runtime",ascending=True).reset_index().iloc[:n]



# For the worst IMDb average we droped the worst two because they had an average of 0 and it´s not considered representative



fig = px.bar(data1,x=data1["Directors"],y=data1["Runtime"],color="Runtime",

             title=f"Top {n} Directors with longest average runtime for directed movies")

fig.show()



fig = px.bar(data2,x=data2["Directors"],y=data2["Runtime"],color="Runtime",

            title=f"Top {n} Directors with shortest average runtime for directed movies")

fig.show()
# How many different genres do we have?





print(f" We have {df['Genres'].nunique()} different genres")

print("\n")

print(df["Genres"].value_counts())
# Some films cover various genres

# We can extract these genres by separating the words where the is a ","



all_genres = []



for genre in df["Genres"].dropna():

    movie_genres = genre.split(",")

    for i in movie_genres:

        all_genres.append(i)
d = Counter(all_genres)

print("These are the basic genres:")

d
# Let´s turn this into a dataframe so that we can plot it



genres = pd.DataFrame.from_dict(d, orient='index').reset_index()

genres["Genre"] = genres["index"]

genres["Count"] = genres[0]

genres.drop(["index",0],axis=1,inplace=True)

genres.head()
genres.info()
# Genres



n = 30



# 1. Genre distribution as described in dataset. Including mixed genres.



data1 = df.groupby(by="Genres").count().sort_values(by="ID",ascending=False).reset_index().iloc[:n]



fig = px.pie(data1,names=data1["Genres"],values=data1["ID"],

            title=f'General Genre Distribution. Top {n} most common movie genres')

fig.show()



# 2. Genre individual distribution.

# If we split up the movies with a mixed Genre (i.e: Comedy,Drama,Romance)

# we get 27 basic genres.



fig = px.pie(genres, values='Count', names='Genre', 

             title='Movie Genres Proportion')

fig.show()
mixed_genres = []



for i in df["Genres"].dropna():

    if "," in i:

        mixed_genres.append(i)



d = Counter(mixed_genres)



print(f"We found a total of {len(d)} different genres")



# Create dataframe



df_mixedg = pd.DataFrame.from_dict(d, orient='index').reset_index()

df_mixedg["Genre"] = df_mixedg["index"]

df_mixedg["Count"] = df_mixedg[0]

df_mixedg.drop(["index",0],axis=1,inplace=True)

df_mixedg.head()
n = 30



# 3. Most common mixed genres.



fig = px.pie(df_mixedg.sort_values(by="Count",ascending=False).reset_index().iloc[:n], values='Count', names='Genre', 

             title=f"Top {n} most common movie mixed genres")

fig.show()
df_genre_year = pd.DataFrame(index=range(1902,2021),columns=Counter(all_genres).keys())

#df_genre_year.astype("float64")

df_genre_year.head()
for year in range(1902,2021):

    genre_year = []

    for genres in df[df["Year"]==year]["Genres"].dropna():

        for i in genres.split(","):

            genre_year.append(i)



    d1 = Counter(genre_year)

    

    if len(d)==0:

        #df_genre_year.loc[year].fillna(0,inplace=True)

        pass

    

    df_genre_year.loc[year] = pd.Series(d1)



# Check some random years 

    

df_genre_year.loc[1928:1933]
# Fill all the missing values with 0



df_genre_year.fillna(0,inplace=True)

df_genre_year.loc[1928:1933]
df_genre_year.plot(figsize=(14,10))



# I tried to make this interactive but for some reason it doesn´t work in Kaggle... This is a bit messier but we can see that Drama has pretty much always been the most popular genre.
n = 100



text = ",".join(word for word in df["Title"])

wordcloud = WordCloud(max_words=n,collocations=False,background_color="white").generate(text)

plt.figure(figsize=(15,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))



print(f"Top {n} most common words used in the movie titles 1902-2020:")

plt.show()
year1 = 1950

year2 = 2019

n = 100



text = ",".join(word for word in df[df["Year"]==year1]["Title"])

wordcloud = WordCloud(max_words=n,collocations=False,background_color="white").generate(text)

plt.figure(figsize=(15,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))



print(f"Top {n} most common words used in the movie titles in {year1}:")

plt.show()



text = ",".join(word for word in df[df["Year"]==year2]["Title"])

wordcloud = WordCloud(max_words=n,collocations=False,background_color="white").generate(text)

plt.figure(figsize=(15,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))



print("\n")

print(f"Top {n} most common words used in the movie titles in {year2}:")

plt.show()