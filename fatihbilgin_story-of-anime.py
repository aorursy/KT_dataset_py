import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import datetime

import seaborn as sns

import matplotlib.pyplot as plt

import plotly as py

import plotly.express as px

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) 

from wordcloud import WordCloud

import warnings

warnings.filterwarnings('ignore')









df_anime = pd.read_csv('../input/anime_filtered.csv')

df_users = pd.read_csv('../input/users_filtered.csv')

df_userlists = pd.read_csv('../input/animelists_filtered.csv')
df_anime.info()
df_users.info()
df_userlists.info()
df_anime.head(3)
df_users.head(3)
df_userlists.head(3)
anime = df_anime[df_anime.genre.notnull()][['anime_id','title','type','source','score','scored_by','rank','popularity','genre']]

users = df_users[df_users.gender.isin(['Female','Male'])][['username','gender','user_completed','user_days_spent_watching','birth_date' ]]

userlists = df_userlists[df_userlists.my_status.isin([1,2]) & df_userlists.anime_id.notnull()][['username', 'anime_id', 'my_score', ]]



userlists = pd.merge(userlists,users, how='inner')

userlists = pd.merge(userlists,anime, how='left')



userlists_sub = userlists[userlists.genre.notnull()].head(100000)

userlists_sub.head()
anime_rank_100 = df_anime[df_anime.popularity!=0].sort_values(by='rank').head(100).loc[:,['popularity','rank', 'title','type', 'source', 'scored_by','favorites','score']]

popularity_and_rank_100 = anime_rank_100[(anime_rank_100.popularity <= 100)]

popularity_and_rank_100["point"] = (popularity_and_rank_100["scored_by"] * popularity_and_rank_100["favorites"] * popularity_and_rank_100["score"]) / 10000000000

popularity_and_rank_100
data = [

    {

        'y':popularity_and_rank_100["popularity"],

        'x': popularity_and_rank_100["rank"],

        'mode': 'markers',

        'marker': {

            'color': popularity_and_rank_100["popularity"],

            'size':  popularity_and_rank_100["point"],

            'showscale': True,

            'sizemin':4

        },

        "text" :  popularity_and_rank_100["title"]

    }

]



layout = go.Layout(title='In Terms Of Rank And Popularity TOP 100 Animes',

                   xaxis=dict(title='Rank'),

                   yaxis=dict( title='Popularity'),

                   autosize=False,

                   width=800,

                   height=600

)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
df_title = popularity_and_rank_100['title'].astype("str").tolist()

df_point  = popularity_and_rank_100['point'].astype("int32").tolist()



list_popularity_and_rank_100 = []



for i in range(0, len(df_point)):

    for j in range(0, df_point[i]):

        list_popularity_and_rank_100.append(df_title[i])



list_popularity_and_rank_100[-10:]
plt.subplots(figsize=(14,7))

wordcloud = WordCloud(    collocations=False,

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(list_popularity_and_rank_100))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
anime_sorted_by_pop = df_anime[df_anime.popularity!=0].sort_values(by='popularity').head(100).loc[:,['title','popularity','rank']]

anime_sorted_by_pop.head()
f,ax1 = plt.subplots(figsize =(25,5))

sns.pointplot(x='popularity',y='rank',data=anime_sorted_by_pop,color='red')

plt.grid()
anime_sorted_by_pop[anime_sorted_by_pop["popularity"]==17]
anime_genre = anime.genre

anime_genre.head()
genre_list = []



genre_splited = []



for i in anime_genre.index:

    for j in anime_genre[i].split(", "):

        genre_splited.append(j)

        if j not in genre_list:

            genre_list.append(j)        
genre_splited[0:6]
anime_genres_count = pd.Series(genre_splited).value_counts() 



plt.figure(figsize=(15,10))

sns.barplot(x=anime_genres_count.index.tolist(), y=anime_genres_count.tolist())

plt.xlabel('Genres')

plt.ylabel('Anime Count')

plt.title('The Most Popular Genres In The Anime Industry (Regarded all Multi-label Genre Tags)') 

plt.xticks(rotation= 75) 

plt.show()
genre_firsts = []



for i in anime_genre.index:

    genre_firsts.append(anime_genre[i].split(", ")[0])
anime_genres_firsts = pd.Series(genre_firsts).value_counts()   



plt.figure(figsize=(10,12))

sns.barplot(x=anime_genres_firsts[0:25].tolist(), y=anime_genres_firsts[0:25].index.tolist())

plt.xlabel('Genres')

plt.ylabel('Anime Count')

plt.title('25 Of The Most Popular Genres In The Anime Industry (Considered First Genre Tag)') 

plt.show()
genres_with_comedy = []



for i in anime_genre.index:

    if anime_genre[i].find('Comedy') > -1:

        for j in anime_genre[i].split(", "):

            if j != 'Comedy':

                genres_with_comedy.append(j)       
genres_with_comedy_count = pd.Series(genres_with_comedy).value_counts().head(10)   



fig = {

  "data": [

    {

      "values": genres_with_comedy_count.tolist(),

      "labels": genres_with_comedy_count.index.tolist(),

      "domain": {"x": [0, .8]},

      "name": "Number Of Students Rates",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    },],

  "layout": {

        "title":"Top 10 Multi-label Tags With Comedy"

    }

}

iplot(fig)
genre_one_label = []



for i in anime_genre.index:

    if len(anime_genre[i].split(", ")) == 1:

        genre_one_label.append(anime_genre[i])         
anime_genres_one_label_count = pd.Series(genre_one_label).value_counts()  



plt.subplots(figsize=(14,7))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(anime_genres_one_label_count.index.tolist()))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
# for 43 genre

F = [0]*43

M =   [0]*43



genre_df = pd.DataFrame({'genre': genre_list, 'Female': F, 'Male': M})

genre_df.set_index('genre', inplace=True)



for i in userlists_sub.index:

    for j in userlists_sub.genre[i].split(", "):

        genre_df[userlists_sub.gender[i]][j] +=1 
genre_df.head()
#genre

Female = []

Male = []



for i in genre_list:

    Female.append(genre_df.loc[i,'Female']/sum(genre_df.loc[i,:]))

    Male.append(genre_df.loc[i,'Male']/sum(genre_df.loc[i,:]))  



f,ax = plt.subplots(figsize=(8,16))

sns.barplot(x=Female, y=genre_list, label='Female', color='r', alpha = 0.7)

sns.barplot(x=Male, y=genre_list, label='Male', color='b', alpha = 0.4)



ax.set(xlabel='Percentage of Genders', ylabel='Genres', title='Percentage of Anime Genres According to Gender')

ax.legend(loc='lower right',frameon= True)

plt.show()
users['birth_date'] = pd.to_datetime(users['birth_date'], errors = 'coerce')

users=users[users.birth_date.notnull()]



birth_date = users.birth_date

gender = users.gender

spent = users.user_days_spent_watching

 

age = []

for each in birth_date:

    age.append(round((datetime.datetime.now()-each).days/365.25,1))
age_dict = {'gender':gender,'age':age,'spent':spent}

users_age_spent = pd.DataFrame(age_dict,columns=['gender','age','spent'])



fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(5, 6))

sns.boxplot(x="gender", y="age", data=users_age_spent, palette="Set1", ax=ax)

ax.set_ylim([0, 60])

plt.show()
age.sort()

age[0:5]



#in the first step i dropped obvious outliers



users_age_spent.drop(users_age_spent[users_age_spent.spent>1000].index, inplace=True)

users_age_spent.drop(users_age_spent[users_age_spent.age>80].index, inplace=True)
users_age_spent.head()
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(5, 6))

sns.boxplot(x="gender", y="spent", data=users_age_spent, palette="Set1", ax=ax)

ax.set_ylim([0, 300])

plt.show()
users_age_spent.plot(kind='scatter', x='age', y='spent', alpha=0.5, figsize = (15,9),

                     color=["r" if each =="Female" else "b" for each in users_age_spent.gender])

plt.show()
F_stats = df_users[(df_users["gender"]=='Female') & (df_users["stats_episodes"]>=0)].loc[:,["stats_rewatched","stats_mean_score"]]     

M_stats = df_users[(df_users["gender"]=='Male') & (df_users["stats_episodes"]>=0)].loc[:,["stats_rewatched","stats_mean_score"]]    



f,ax=plt.subplots(1,2,figsize=(14,6))

F_stats.stats_mean_score.plot.hist(ax=ax[0],bins=10,edgecolor='black',color='mediumvioletred')

ax[0].set_title('Female Mean Score')

x1=list(range(0,11,1))

ax[0].set_xticks(x1)

M_stats.stats_mean_score.plot.hist(ax=ax[1],bins=10,edgecolor='black',color='slateblue')

ax[1].set_title('Male Mean Score')

x2=list(range(0,11,1))

ax[1].set_xticks(x2)

plt.suptitle('Anime Rating by Gender')

plt.show()
anime_premiered = df_anime[df_anime.premiered.notnull()].premiered

release_seasons = []



for i in anime_premiered.index:

    release_seasons.append(anime_premiered[i].split(" ")[0])



apc = pd.Series(release_seasons).value_counts()   

anime_premiered_count = pd.DataFrame({'season':apc.index, 'premier':apc.values, 'order':[1,3,4,2]})

anime_premiered_count.set_index('order', inplace=True)

anime_premiered_count.sort_index(inplace=True)
anime_premiered_count
plt.figure(figsize=(5,4))

sns.barplot(x='season',y='premier', data=anime_premiered_count, palette="muted")

plt.xticks(rotation=60)

plt.title('Animes Released on the Premiere Seasons',color = 'darkred',fontsize=12)

plt.show()
anime_premiered = df_anime[df_anime.premiered.notnull()].premiered

premier_seasons = []

premier_years = []





for i in anime_premiered.index:

    if(2009 <= int(anime_premiered[i].split(" ")[1]) <= 2018):

        premier_seasons.append(anime_premiered[i].split(" ")[0])

        premier_years.append(anime_premiered[i].split(" ")[1])

        

premier_df = pd.DataFrame({

                            'Year':premier_years,

                            'Spring': [1 if each == 'Spring' else 0 for each in premier_seasons],

                            'Summer': [1 if each == 'Summer' else 0 for each in premier_seasons],

                            'Fall':   [1 if each == 'Fall'   else 0 for each in premier_seasons],

                            'Winter': [1 if each == 'Winter' else 0 for each in premier_seasons],

                            })



premier_years_seasons = premier_df.groupby(['Year'])['Spring','Summer','Fall','Winter'].agg('sum')
premier_years_seasons.plot(kind='bar',stacked=True, figsize=(10,5), title='Animes Released on the Premiere Seasons by Years (Last 10 year)')

plt.show()
broadcast = [x.split(" at ")[0].strip() for x in df_anime["broadcast"].astype("str")]

broad_days = pd.Series(broadcast).value_counts()[2:9]

df_broad_days = pd.DataFrame({'days':broad_days.index, 'broadcast':broad_days.values, 'index':[7,6,5,2,4,1,3]}).set_index('index').sort_index()



plt.figure(figsize=(7,4))

sns.barplot(x='days',y='broadcast', data=df_broad_days, palette="hls")

plt.xticks(rotation=60)

plt.title('Animes Released on the Premiere Seasons',color = 'darkred',fontsize=14)

plt.show()
animetypes = df_anime['type'].value_counts(dropna=False)
print(animetypes)
plt.figure(figsize=(8,5))

sns.barplot(x=animetypes.index,y=animetypes.values, palette="rocket")

plt.title('Types of Animes',color = 'purple',fontsize=15)

plt.show()
animesources= df_anime['source'].value_counts(dropna=False)

animesources = animesources[animesources.index != 'Unknown']
animesources.head(5)
plt.figure(figsize=(10,6))

sns.barplot(x=animesources.index,y=animesources.values, palette="Blues_d")

plt.xticks(rotation=60)

plt.title('Sources of Animes',color = 'darkblue',fontsize=15)

plt.show()
df_anime_ghibli = df_anime[(df_anime.favorites >= 500) & (df_anime["rank"] <= 500) & (df_anime.title_english.notnull()) & 

                           (df_anime.studio == 'Studio Ghibli') & (df_anime.type == 'Movie')]

anime_ghibli = df_anime_ghibli.loc[:,["title_english", "favorites", "aired_string", "rank", "score"]]
#i normalized the result with p_norm

anime_ghibli["p_norm"] = ((anime_ghibli["favorites"] - np.min(anime_ghibli["favorites"]))) / (np.max(anime_ghibli["favorites"]) - np.min(anime_ghibli["favorites"]))

#i added 000001 to showed The Tale of the Princess Kaguya which has 0 p_norm.

anime_ghibli["point"] = (anime_ghibli["p_norm"]+0.000001)*75

anime_ghibli["year"] = [x.split(", ")[1] for x in anime_ghibli["aired_string"]]

anime_ghibli.loc[2258:2259,"title_english"] = "Howl's Moving Castle"
anime_ghibli.sort_values("rank")
data = [

    {

        'y':anime_ghibli["score"],

        'x': anime_ghibli["rank"],

        'mode': 'markers',

        'marker': {

            'color':anime_ghibli["score"],

            'size':  anime_ghibli["point"],

            'showscale': True,

            'sizemin':3

        },

        "text" : anime_ghibli["title_english"] + ' (' + anime_ghibli["year"] + ')'

    }

]



layout = go.Layout(title='Popular Anime Movies From Studio Ghibli',

                   xaxis=dict(title='Rank'),

                   yaxis=dict( title='Score'),

                   autosize=False,

                   width=800,

                   height=600

)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
anime_movies = df_anime[(df_anime.type == "Movie") & (df_anime.aired_string != "Not available")]

anime_movies["year"] = [x.split(",")[1].strip()[0:4] if len(x.split(",")) > 1 else x.split(",")[0].strip()[0:4] for x in anime_movies["aired_string"]]

#location = [x.split(",")[1].strip() if len(x.split(",")) > 1 else x.split(",")[0].strip() for x in df_users["location"].astype("str")]

anime_movies = anime_movies["year"].value_counts()

anime_movies_years = pd.DataFrame({'Year':anime_movies.index, 'Movie Count':anime_movies.values})

anime_movies_years = anime_movies_years.sort_values("Year")

#i think 2018 and 2019 is incomplete

anime_movies_years = anime_movies_years.iloc[:-2,:]
fig = px.line(anime_movies_years, x="Year", y="Movie Count", 

              title='Anime Movies by Years', width=750, height=400)

fig.show()
location = [x.split(",")[1].strip() if len(x.split(",")) > 1 else x.split(",")[0].strip() for x in df_users["location"].astype("str")]
pd.Series(location).value_counts()[:5] 
location = ["USA" if x in ("USA", "United States of America", "U.S.A.", "US", "U.S.", "U.S.A", 

                           "usa", "United States", "California", "Texas", "New York", "Florida", 

                           "Ohio", "Michigan", "Illinois", "Washington", "Pennsylvania", "Virginia", 

                           "Arizona", "Maryland", "Tennessee",  "New Jersey", "North Carolina", 

                           "Colorado", "Oregon", "Indiana", "Massachusetts", "Minnesota", "NY", "CA", 

                           "TX", "FL", "PA", "IL", "MA", "NC", "NJ", "AZ", "GA", "VA", "IN","TN", 

                           "MI", "SC", "MN", "NYC", "MD", "MO", "WI", "Tx", "CT", "NV", "OR", "KY", 

                           "OH", "Ca", "CO", "LA", "DC", "AL", "ny", "NH", "Philadelphia", "north carolina", 

                           "Missouri", "Nevada", "Kentucky", "Louisiana", "Connecticut", "california", 

                           "Oklahoma", "Alabama", "Hawaii", "Kansas", "Utah", "Iowa", "South Carolina", 

                           "Arkansas", "Nebraska", "texas", "Southern California", "New England", 

                           "Mississippi", "florida", "new york", "Idaho", "New Mexico", "Chicago", 

                           "New Hampshire", "Los Angeles", "Rhode Island", "New York City", "Maine", 

                           "America", "Alaska", "Delaware", "Northern Ireland", "ohio", "Seattle", 

                           "West Virginia", "North Dakota", "South Dakota", "Boston", "Vermont", "Montana", 

                           "michigan", "washington", "New york", "Las Vegas", "Wisconsin", 

                           "Washington State", "SoCal")  else x for x in location]

location = ["UK" if x in ("England", "england", "United Kingdom", "Scotland", "Glasgow", "London", "london", 

                          "Wales", "Manchester", "Nottingham", "Kent", "England.", "united kingdom", "Essex", 

                          "UK", "Uk", "uk", "Britain", "Liverpool", "Birmingham") else x for x in location]

location = ["Netherlands" if x in ("Netherlands", "The Netherlands", "Nederland", "Netherland", "the Netherlands", 

                                   "The netherlands", "netherlands", "Holland", "Amsterdam", "Utrecht", "Rotterdam", 

                                   "Limburg", "the netherlands", "Zuid-Holland") else x for x in location]

location = ["Canada" if x in ("Canada", "canada", "CANADA", "Ontario", "ON", "Quebec", "Québec", "Alberta", 

                              "Toronto", "Vancouver", "British Columbia", "Saskatchewan", "ontario", "BC", 

                              "Montreal", "Manitoba", "Nova Scotia") else x for x in location]

location = ["Brazil" if x in ("Brazil", "Brasil", "São Paulo", "Sao Paulo", "São Paulo - Brazil", "Paraná", 

                              "Pará", "Rio de Janeiro", "RJ", "RS", "SP", "MG", "DF", "Porto Alegre", 

                              "Minas Gerais", "brasil", "Brasil.", "Rio Grande do Sul", "Santa Catarina", 

                              "Bahia", "Ceará", "Pernambuco", "Brazil.", "brazil", 

                              "Goiás") else x for x in location]

location = ["Russia" if x in ("Russia", "Moscow", "Russian Federation", "Saint-Petersburg", "St. Petersburg", 

                              "Saint Petersburg", "Novosibirsk", "Россия", 

                              "St.Petersburg") else x for x in location]

location = ["Japan" if x in ("Japan", "japan", "Tokyo", "tokyo", "Ikebukuro", "Kyoto", 

                             "Osaka") else x for x in location]

location = ["Turkey" if x in ("Turkey", "Istanbul", "İstanbul", "istanbul", "Ankara", "Bursa", "Türkiye", 

                              "Turkiye", "turkey", 

                              "İzmir", "Izmir", "Antalya", "TURKEY") else x for x in location]

location = ["Philippines" if x in ("Philippines", "philippines", "Manila", "Cavite", "Phillipines", 

                                   "Metro Manila", "manila", "Cebu", "Laguna", 

                                   "Quezon City") else x for x in location]

location = ["Indonesia" if x in ("Indonesia", "Jakarta", "indonesia", "Bandung", "West Java", "Central Java", 

                                 "jakarta", "Jawa Timur", "Banten", "Jawa Barat", "East Java", "Indonesian", 

                                 "Yogyakarta") else x for x in location]

location = ["Spain" if x in ("Spain", "España", "Barcelona", "Madrid", "Valencia", 

                             "Catalonia") else x for x in location]

location = ["Poland" if x in ("Poland", "poland", "Warsaw", "Warszawa", "Poznań", "Łódź", "Gdańsk", "Wrocław", 

                              "Kraków", "Cracow", "Szczecin", "Gdynia", "Bydgoszcz", "Lublin", "Białystok", 

                              "Katowice", "Rzeszów", "Lodz") else x for x in location]

location = ["France" if x in ("France", "Paris", "france", "paris", "FRANCE") else x for x in location]

location = ["Australia" if x in ("Australia", "australia", "Sydney", "Melbourne", "Victoria", "WA", "NSW",

                                 "Western Australia", "New South Wales", "Adelaide", "Queensland", "Perth",

                                 "South Australia", "Brisbane") else x for x in location]

location = ["Portugal" if x in ("Portugal", "Lisbon", "Porto", "Lisboa", "portugal") else x for x in location]

location = ["Italy" if x in ("Italy", "Italia", "italy", "Rome", "Roma", "italia") else x for x in location]

location = ["Mexico" if x in ("Mexico", "México", "Mexico City", "mexico", "Baja California", 

                              "Jalisco") else x for x in location]

location = ["Argentina" if x in ("Argentina", "Buenos Aires") else x for x in location]

location = ["Greece" if x in ("Greece", "Athens", "greece", "athens") else x for x in location]

location = ["Hungary" if x in ("Hungary", "Budapest") else x for x in location]

location = ["India" if x in ("India", "india", "INDIA", "Mumbai") else x for x in location]

location = ["Sweden" if x in ("Sweden", "sweden", "Stockholm", "Gothenburg") else x for x in location]

location = ["Latvia" if x in ("Latvia", "Riga") else x for x in location]

location = ["Germany" if x in ("Germany", "Berlin", "Deutschland", "NRW", "Hamburg", "germany", 

                               "Bavaria", "Hessen") else x for x in location]

location = ["Malaysia" if x in ("Malaysia", "malaysia", "Kuala Lumpur", "Selangor", 

                                "Sarawak") else x for x in location]

location = ["Bulgaria" if x in ("Bulgaria", "Sofia") else x for x in location]

location = ["Singapore" if x in ("Singapore", "singapore") else x for x in location]

location = ["Romania" if x in ("Romania", "Bucharest") else x for x in location]

location = ["Austria" if x in ("Austria", "Vienna") else x for x in location]

location = ["Israel" if x in ("Israel", "israel") else x for x in location]

location = ["Lithuania" if x in ("Lithuania", "Vilnius", "Kaunas") else x for x in location]

location = ["Czech Republic" if x in ("Czech Republic", "Czech republic", "Prague") else x for x in location]

location = ["Estonia" if x in ("Estonia", "Tallinn") else x for x in location]

location = ["Ukraine" if x in ("Ukraine", "Kiev", "Odessa") else x for x in location]

location = ["Norway" if x in ("Norway", "Oslo", "norway") else x for x in location]

location = ["Colombia" if x in ("Colombia", "Cali") else x for x in location]

location = ["New Zealand" if x in ("New Zealand", "Auckland", "NZ") else x for x in location]

location = ["Finland" if x in ("New Zealand", "Helsinki") else x for x in location]

location = ["Belgium" if x in ("Belgium", "belgium", "Antwerp") else x for x in location]

location = ["China" if x in ("China", "Hong Kong") else x for x in location]

location = ["Vietnam" if x in ("Vietnam", "Viet Nam") else x for x in location]

location = ["Peru" if x in ("Peru", "Perú", "Lima") else x for x in location]

location = ["Saudi Arabia" if x in ("Saudi Arabia", "Riyadh", "KSA", "K.S.A", "Jeddah", "saudi arabia", 

                                    "jeddah") else x for x in location]

location = ["Kuwait" if x in ("Kuwait", "kuwait") else x for x in location]

location = ["Thailand" if x in ("Thailand", "Bangkok") else x for x in location]

location = ["Bangladesh" if x in ("Bangladesh", "Dhaka", "bangladesh") else x for x in location]

location = ["United Arab Emirates" if x in ("UAE", "Dubai") else x for x in location]

location = ["Ireland" if x in ("Ireland", "Dublin") else x for x in location]

location = ["Chile" if x in ("Chile", "Santiago") else x for x in location]

location = ["Serbia" if x in ("Serbia", "Belgrade") else x for x in location]

location = ["Egypt" if x in ("Egypt", "Cairo") else x for x in location]

location = ["Belarus" if x in ("Belarus", "Minsk") else x for x in location]

location = ["Denmark" if x in ("Denmark", "Copenhagen") else x for x in location]

location = ["South Korea" if x in ("South Korea", "Korea", "Seoul") else x for x in location]

location = ["Croatia" if x in ("Croatia", "Zagreb") else x for x in location]

location = ["Georgia" if x in ("Georgia", "Tbilisi", "georgia") else x for x in location]

#Georgia can be state of America "Georgia" or country "Georgia". I putted it in country category. 

#And i putted GA in state category.
pd.Series(location).value_counts()[:5] 
loc = pd.Series(location).value_counts()[:25] 

df_loc = pd.DataFrame({'Country':loc.index, 'Members':loc.values})



fig = px.bar(df_loc, x='Country', y='Members', title="Anime Watchers Around The World (MyAnimeList Members)")

fig.show()
loc_exc_usa = pd.Series(location).value_counts()[0:250] 

df_loc_exc_usa = pd.DataFrame({'Country':loc_exc_usa.index, 'Members':loc_exc_usa.values})

df_loc_exc_usa = df_loc_exc_usa[df_loc_exc_usa.Country != "Antarctica"].reset_index(drop=True)



data = [ dict(

        type = 'choropleth',

        locations = df_loc_exc_usa['Country'],

        locationmode = 'country names',

        z = df_loc_exc_usa['Members'],

        text = df_loc_exc_usa['Country'],

        colorscale=

            [[0.0, "rgb(251, 237, 235)"],

            [0.09, "rgb(245, 211, 206)"],

            [0.12, "rgb(239, 179, 171)"],

            [0.15, "rgb(236, 148, 136)"],

            [0.22, "rgb(239, 117, 100)"],

            [0.29, "rgb(235, 90, 70)"],

            [0.36, "rgb(207, 81, 61)"],

            [0.41, "rgb(176, 70, 50)"],

            [0.77, "rgb(147, 59, 39)"],

            [1.00, "rgb(110, 47, 26)"]],

        autocolorscale = False,

        reversescale = False,

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) ),

        colorbar = dict(

            autotick = False,

            tickprefix = '',

            title = 'Members'),

      ) ]



layout = dict(

    title = 'Anime Watchers Around The World (On The Basis Of MyAnimeList)',

    geo = dict(

        showframe = False,

        showcoastlines = False,

        projection = dict(

            type = 'Mercator'

        )

    )

)



w_map = dict( data=data, layout=layout )

iplot( w_map, validate=False)