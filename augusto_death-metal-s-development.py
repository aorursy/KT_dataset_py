# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
bands = pd.read_csv(

    filepath_or_buffer="../input/bands.csv", 

    sep=",", 

    na_values=["N/A"], 

    dtype={"id": "int32", "formed_in": "float32"}

)
albums = pd.read_csv(

    filepath_or_buffer="../input/albums.csv", 

    sep=",", 

    na_values=["N/A"], 

    dtype={"id": "int32", "band": "int32", "year": "int32"}

)
first_death_metal_band = bands.loc[bands.formed_in.idxmin()]

print("The world's first death metal band is \"{}\" which formed in {:.0f}.".format(first_death_metal_band["name"], first_death_metal_band["formed_in"]))
for key, value in zip(first_death_metal_band.index.values, first_death_metal_band):

    print("{:>10}: {:<}".format(key, value))
import matplotlib.pyplot as plt

plt.style.use('ggplot')



bands_count = bands.groupby("formed_in")["id"].count().cumsum().loc[:2015]

albums_count = albums.groupby("year")["id"].count().cumsum().loc[:2015]



df = pd.DataFrame(

    {

        "bands": bands_count,

        "albums": albums_count

    }

)



ax = df.plot(marker="h")

_ = ax.set_xlabel("year")

_ = ax.set_ylabel("acum number")
bands_count = bands.groupby("formed_in")["id"].count().loc[:2015]

albums_count = albums.groupby("year")["id"].count().loc[:2015]



df = pd.DataFrame(

    {

        "bands": bands_count,

        "albums": albums_count

    }

)



ax = df.plot(marker="h")

_ = ax.set_xlabel("year")

_ = ax.set_ylabel("acum number")

_ = ax.annotate("Nirvana's \"nevermind\" was released", xy=(1991, 950), xytext=(1980, 1600), arrowprops={"width": 1.0})
ax = bands[["formed_in", "genre"]].drop_duplicates().groupby("formed_in")["genre"].count().plot(color="#ee7621", marker="h", ylim=[0, 400])

_ = ax.set_xlabel("year")

_ = ax.set_ylabel("number of genres of new bands")
dominant_genres = bands.groupby("genre")["genre"].count().sort_values().tail(10)

dominant_genres["others"] = bands.shape[0] - dominant_genres.sum()

_ = dominant_genres.plot.pie(figsize=(6, 6))
genre_count = bands.groupby("genre")["id"].count()

main_genres = genre_count[genre_count >= 50].index.values

main_genres = [genre for genre in main_genres if "/" not in genre]



main_genres_bands = bands[bands.genre.isin(main_genres)]

main_genres_albums = pd.merge(

    left=main_genres_bands,

    right=albums,

    left_on="id",

    right_on="band",

    suffixes=["_band", "_album"],

    how="inner"

)[["id_album", "year", "genre"]]



main_genres_albums = main_genres_albums.groupby(["year", "genre"])["id_album"].count().unstack("genre").fillna(0)

main_genres_albums = main_genres_albums.loc[:2015] # data of 2017 is incomplete.

main_genres_albums.drop(["Blackened Death Metal", "Industrial Death Metal", "Experimental Death Metal"], axis=1, inplace=True)



_ = main_genres_albums.plot(figsize=(8, 5), marker="h")
bands_albums = pd.merge(

    left=bands, 

    right=albums, 

    left_on="id", 

    right_on="band", 

    suffixes=["_band", "_album"],

    how="left"

).drop("band", axis=1)
bands_albums_count = pd.DataFrame(bands_albums.groupby("id_band")["id_album"].count().sort_values().tail(20))

bands_albums_count.columns = ["albums_count"] 



bands_high_production_top20 = pd.merge(

    left=bands,

    right=bands_albums_count,

    left_on="id",

    right_index=True

)[["name", "albums_count"]].sort_values("albums_count").set_index("name")



_ = bands_high_production_top20.plot.barh(color="#00304e")
reviews = pd.read_csv(

    filepath_or_buffer="../input/reviews.csv", 

    sep=",", 

    na_values=["N/A"], 

    usecols=["id", "album", "title", "score"],

    dtype={"id": "int32", "album": "int32", "score": "float32"}

)
bands_albums_reviews = pd.merge(

    left=bands_albums, 

    right=reviews, 

    left_on="id_album", 

    right_on="album", 

    suffixes=["", "_review"],

    how="left"

).drop("album", axis=1)
albums_reviews_count = pd.DataFrame(bands_albums_reviews.groupby("id_album")["id"].count().sort_values().tail(20))

albums_reviews_count.columns = ["reviews_count"] 



reviews_more_reviews_top20 = pd.merge(

    left=bands_albums,

    right=albums_reviews_count,

    left_on="id_album",

    right_index=True

)[["name", "title", "reviews_count"]].sort_values("reviews_count")



reviews_more_reviews_top20["band/album"] = reviews_more_reviews_top20.name + "'s \"" + reviews_more_reviews_top20.title + "\""

_ = reviews_more_reviews_top20[["band/album", "reviews_count"]].set_index("band/album").plot.barh(color="#00304e", legend=False, figsize=(8, 6))
import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls
country_bands_count = bands.groupby("country")["id"].count()

country_bands_count = pd.DataFrame(country_bands_count)

country_bands_count.columns = ["bands_count"]

country_bands_count.reset_index(inplace=True)
# I copy and modify this piece of code from Anisotropico's kernel 

# "Interactive Plotly: Global Youth Unemployment". Thank him.



metricscale = [

    [0, 'rgb(102,194,165)'], 

    [0.05, 'rgb(102,194,165)'], 

    [0.15, 'rgb(171,221,164)'], 

    [0.2, 'rgb(230,245,152)'], 

    [0.25, 'rgb(255,255,191)'], 

    [0.35, 'rgb(254,224,139)'], 

    [0.45, 'rgb(253,174,97)'], 

    [0.55, 'rgb(213,62,79)'], 

    [1.0, 'rgb(158,1,66)']

]



data = [ dict(

        type = 'choropleth',

        autocolorscale = False,

        colorscale = metricscale,

        showscale = True,

        locations = country_bands_count['country'].values,

        z = country_bands_count['bands_count'].values,

        locationmode = 'country names',

        text = country_bands_count['country'].values,

        marker = dict(

            line = dict(color = 'rgb(250,250,225)', width = 0.5)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = 'Number of Death Metal Bands')

            )

       ]



layout = dict(

    title = 'World Map of Number of Death Metal Bands',

    geo = dict(

        showframe = True,

        showocean = True,

        oceancolor = 'rgb(0,0,52)',

        #oceancolor = 'rgb(222,243,246)',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = False,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = False,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )

fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='worldmapdeathmetal')
ax = bands.groupby("country")["id"].count().sort_values(ascending=False).head(20).sort_values().plot(kind="barh", color="#00304e")

_ = ax.set_xlabel("number of bands")
albums_reviews_count = bands_albums_reviews.groupby("id_album")["id"].count()

popular_albums = albums_reviews_count[albums_reviews_count > 10].index.values

popular_bands_albums_reviews = bands_albums_reviews[bands_albums_reviews.id_album.isin(popular_albums)]



best_albums = popular_bands_albums_reviews.groupby("id_album")["score"].sum() / popular_bands_albums_reviews.groupby("id_album")["score"].count()



best_albums = pd.DataFrame(best_albums.sort_values().tail(50))

best_albums.columns = ["average_score"] 



bands_albums_best_top = pd.merge(

    left=bands_albums,

    right=best_albums,

    left_on="id_album",

    right_index=True

)[["name", "title", "average_score"]].sort_values("average_score")



bands_albums_best_top["band/album"] = bands_albums_best_top.name + "'s \"" + bands_albums_best_top.title + "\""

_ = bands_albums_best_top[["band/album", "average_score"]].set_index("band/album").plot.barh(color="#00304e", legend=False, figsize=(8, 12), xlim=[0.85, 1.0])