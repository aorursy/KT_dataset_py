# loading packages

import pandas as pd

import plotly.plotly as py

import plotly

from plotly.graph_objs import *

import plotly.graph_objs as go
# load data

spotify = pd.read_csv(r"C:\Users\KK\Documents\Kitu\College\Senior Year\Extracurriculars\Python\Spotify\.spyproject\data.csv")
# explore data

spotify.head()

spotify.shape

spotify.dtypes
# change variable data types

spotify.Region = spotify.Region.astype("category")

spotify.Date = pd.to_datetime(spotify["Date"])

spotify.dtypes
# look into counts of Region

spotify["Region"].value_counts()

len(spotify["Region"].value_counts())
# make new dataframes for global, usa, great britain, mexico, taiwan, and singapore

globe = spotify[spotify.Region == "global"]

usa = spotify[spotify.Region == "us"]

great_britain = spotify[spotify.Region == "gb"]

mexico = spotify[spotify.Region == "mx"]

taiwan = spotify[spotify.Region == "tw"]

singapore = spotify[spotify.Region == "sg"]
# create descending table of top songs by total stream count per country

# also add new country variable for use in merging

top_globe = globe.groupby("Track Name").agg({"Streams": "sum"})

top_globe = top_globe.sort_values(["Streams"], ascending = False)

top_globe["country"] = "Globe"



top_usa = usa.groupby("Track Name").agg({"Streams": "sum"})

top_usa = top_usa.sort_values(["Streams"], ascending = False)

top_usa["country"] = "USA"



top_great_britain = great_britain.groupby("Track Name").agg({"Streams": "sum"})

top_great_britain = top_great_britain.sort_values(["Streams"], ascending = False)

top_great_britain["country"] = "Great Britain"



top_mexico = mexico.groupby("Track Name").agg({"Streams": "sum"})

top_mexico = top_mexico.sort_values(["Streams"], ascending = False)

top_mexico["country"] = "Mexico"



top_taiwan = taiwan.groupby("Track Name").agg({"Streams": "sum"})

top_taiwan = top_taiwan.sort_values(["Streams"], ascending = False)

top_taiwan["country"] = "Taiwan"



top_singapore = singapore.groupby("Track Name").agg({"Streams": "sum"})

top_singapore = top_singapore.sort_values(["Streams"], ascending = False)

top_singapore["country"] = "Singapore"
# add a new variable of the proportion of the song from all streams

top_globe["prop"] = top_globe["Streams"]/sum(top_globe["Streams"])*100

top_usa["prop"] = top_usa["Streams"]/sum(top_usa["Streams"])*100

top_great_britain["prop"] = top_great_britain["Streams"]/sum(top_great_britain["Streams"])*100

top_mexico["prop"] = top_mexico["Streams"]/sum(top_mexico["Streams"])*100

top_taiwan["prop"] = top_taiwan["Streams"]/sum(top_taiwan["Streams"])*100

top_singapore["prop"] = top_singapore["Streams"]/sum(top_singapore["Streams"])*100
# subset to only top 3 songs

top_globe = top_globe[0:3]

top_usa = top_usa[0:3]

top_great_britain = top_great_britain[0:3]

top_mexico = top_mexico[0:3]

top_taiwan = top_taiwan[0:3]

top_singapore = top_singapore[0:3]
# delete Streams variable, since we will be using prop to compare

del top_globe["Streams"]

del top_usa["Streams"]

del top_great_britain["Streams"]

del top_mexico["Streams"]

del top_taiwan["Streams"]

del top_singapore["Streams"]
# row bind all dataframes

top_all_merged = top_globe.append([top_usa, top_great_britain, top_mexico, top_taiwan, top_singapore])
# reset index to include index as a variable

top_all_merged = top_all_merged.reset_index()
# find all unique songs

all_songs = top_all_merged["Track Name"].value_counts()

all_songs = all_songs.reset_index()

len(top_all_merged["Track Name"].value_counts())
# bar plot of top 3 songs by country...interactive via plotly 

# did not use Python's visualizations because they are not as powerful or engaging as plotly



# each trace represents one of nine of the unique songs

# y represents the proportion value for each country

import plotly.plotly as py

from plotly.graph_objs import *



trace1 = {

  "x": ["Global", "USA", "Great Britain", "Mexico", "Taiwan", "Singapore"], 

  "y": [2.54, 1.45, 2.47, 1.75, 2.11, 2.7], 

  "name": "Shape of You", 

  "type": "bar", 

  "uid": "d81641", 

  "visible": True, 

  "xsrc": "sweetmusicality:7:ff97bb", 

  "ysrc": "sweetmusicality:7:593809"

}

trace2 = {

  "x": ["Global", "USA", "Great Britain", "Mexico", "Taiwan", "Singapore"], 

  "y": [1.48, 0, 1.4, 0, 0, 1.48], 

  "name": "Despacito - Remix", 

  "type": "bar", 

  "uid": "c15c84", 

  "visible": True, 

  "xsrc": "sweetmusicality:7:ff97bb", 

  "ysrc": "sweetmusicality:7:724a8c"

}

trace3 = {

  "x": ["Global", "USA", "Great Britain", "Mexico", "Taiwan", "Singapore"], 

  "y": [0, 0, 0, 0, 1.97, 1.78], 

  "name": "Something Just Like This", 

  "type": "bar", 

  "uid": "1dbc1b", 

  "xsrc": "sweetmusicality:7:ff97bb", 

  "ysrc": "sweetmusicality:7:31cdb8"

}

trace4 = {

  "x": ["Global", "USA", "Great Britain", "Mexico", "Taiwan", "Singapore"], 

  "y": [1.3, 0, 0, 2.22, 0, 0], 

  "name": "Despacito (Featuring Daddy Yankee)", 

  "type": "bar", 

  "uid": "c6b042", 

  "xsrc": "sweetmusicality:7:ff97bb", 

  "ysrc": "sweetmusicality:7:4583e2"

}

trace5 = {

  "x": ["Global", "USA", "Great Britain", "Mexico", "Taiwan", "Singapore"], 

  "y": [0, 0, 0, 2.16, 0, 0], 

  "name": "Me Rehúso", 

  "type": "bar", 

  "uid": "be7d95", 

  "xsrc": "sweetmusicality:7:ff97bb", 

  "ysrc": "sweetmusicality:7:b9ea50"

}

trace6 = {

  "x": ["Global", "USA", "Great Britain", "Mexico", "Taiwan", "Singapore"], 

  "y": [0, 1.2, 0, 0, 0, 0], 

  "name": "Mask Off", 

  "type": "bar", 

  "uid": "60d6b8", 

  "xsrc": "sweetmusicality:7:ff97bb", 

  "ysrc": "sweetmusicality:7:989da6"

}

trace7 = {

  "x": ["Global", "USA", "Great Britain", "Mexico", "Taiwan", "Singapore"], 

  "y": [0, 0, 0, 0, 1.23, 0], 

  "name": "演員", 

  "type": "bar", 

  "uid": "f912b1", 

  "xsrc": "sweetmusicality:7:ff97bb", 

  "ysrc": "sweetmusicality:7:cd61ac"

}

trace8 = {

  "x": ["Global", "USA", "Great Britain", "Mexico", "Taiwan", "Singapore"], 

  "y": [0, 0, 1.67, 0, 0, 0], 

  "name": "Castle on the Hill", 

  "type": "bar", 

  "uid": "c01a7b", 

  "xsrc": "sweetmusicality:7:ff97bb", 

  "ysrc": "sweetmusicality:7:083ede"

}

trace9 = {

  "x": ["Global", "USA", "Great Britain", "Mexico", "Taiwan", "Singapore"], 

  "y": [0, 1.51, 0, 0, 0, 0], 

  "name": "HUMBLE.", 

  "type": "bar", 

  "uid": "d9ea4a", 

  "xsrc": "sweetmusicality:7:ff97bb", 

  "ysrc": "sweetmusicality:7:1008dc"

}



# assign the data and create the layout for the barplot

data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9]

layout = {

  "annotations": [

    {

      "x": 1.09648221896, 

      "y": 0.671878877124, 

      "font": {"size": 21}, 

      "showarrow": False, 

      "text": "<b>Song</b>", 

      "xanchor": "middle", 

      "xref": "paper", 

      "yanchor": "bottom", 

      "yref": "paper"

    }

  ], 

  "autosize": True, 

  "barmode": "stack", 

  "font": {"size": 18}, 

  "hovermode": "closest", 

  "legend": {

    "x": 1.01935845381, 

    "y": 0.673239347844, 

    "borderwidth": 0, 

    "orientation": "v", 

    "traceorder": "normal"

  }, 

  "margin": {"b": 80}, 

  "title": "<b>Top 3 Streamed Songs on Spotify from Jan 2017 - Aug 2017 by Country</b>", 

  "titlefont": {"size": 28}, 

  "xaxis": {

    "autorange": False, 

    "domain": [0, 1.01], 

    "range": [-0.5, 5.51343670089], 

    "side": "bottom", 

    "title": "<b>Country</b>", 

    "type": "category"

  }, 

  "yaxis": {

    "anchor": "x", 

    "autorange": False, 

    "domain": [-0.01, 1], 

    "range": [0, 6.66421250763], 

    "title": "<b>% this song was streamed in its country</b>", 

    "type": "linear"

  }

}



# let's plot it!

fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
# create descending table of top artists by total stream count per country

# also add new country variable for use in merging

top_globe2 = globe.groupby("Artist").agg({"Streams": "sum"})

top_globe2 = top_globe2.sort_values(["Streams"], ascending = False)

top_globe2["country"] = "Globe"



top_usa2 = usa.groupby("Artist").agg({"Streams": "sum"})

top_usa2 = top_usa2.sort_values(["Streams"], ascending = False)

top_usa2["country"] = "USA"



top_great_britain2 = great_britain.groupby("Artist").agg({"Streams": "sum"})

top_great_britain2 = top_great_britain2.sort_values(["Streams"], ascending = False)

top_great_britain2["country"] = "Great Britain"



top_mexico2 = mexico.groupby("Artist").agg({"Streams": "sum"})

top_mexico2 = top_mexico2.sort_values(["Streams"], ascending = False)

top_mexico2["country"] = "Mexico"



top_taiwan2 = taiwan.groupby("Artist").agg({"Streams": "sum"})

top_taiwan2 = top_taiwan2.sort_values(["Streams"], ascending = False)

top_taiwan2["country"] = "Taiwan"



top_singapore2 = singapore.groupby("Artist").agg({"Streams": "sum"})

top_singapore2 = top_singapore2.sort_values(["Streams"], ascending = False)

top_singapore2["country"] = "Singapore"
# add a new variable of the proportion of the artist from all streams

top_globe2["prop"] = top_globe2["Streams"]/sum(top_globe2["Streams"])*100

top_usa2["prop"] = top_usa2["Streams"]/sum(top_usa2["Streams"])*100

top_great_britain2["prop"] = top_great_britain2["Streams"]/sum(top_great_britain2["Streams"])*100

top_mexico2["prop"] = top_mexico2["Streams"]/sum(top_mexico2["Streams"])*100

top_taiwan2["prop"] = top_taiwan2["Streams"]/sum(top_taiwan2["Streams"])*100

top_singapore2["prop"] = top_singapore2["Streams"]/sum(top_singapore2["Streams"])*100
# subset to only top 3 artists

top_globe2 = top_globe2[0:3]

top_usa2 = top_usa2[0:3]

top_great_britain2 = top_great_britain2[0:3]

top_mexico2 = top_mexico2[0:3]

top_taiwan2 = top_taiwan2[0:3]

top_singapore2 = top_singapore2[0:3]
# delete Streams variable, since we will be using prop to compare

del top_globe2["Streams"]

del top_usa2["Streams"]

del top_great_britain2["Streams"]

del top_mexico2["Streams"]

del top_taiwan2["Streams"]

del top_singapore2["Streams"]
# row bind all dataframes

top_all_merged2 = top_globe2.append([top_usa2, top_great_britain2, top_mexico2, top_taiwan2, top_singapore2])
# reset index to include index as a variable

top_all_merged2 = top_all_merged2.reset_index()
# find all unique artists

all_artists = top_all_merged2["Artist"].value_counts()

all_artists = all_artists.reset_index()

len(top_all_merged2["Artist"].value_counts())
# bar plot of top 5 artists by country...interactive via plotly 

# did not use Python's visualizations because they are not as powerful or engaging as plotly



# each trace represents one of fourteen of the unique artists

# y represents the proportion value for each country

import plotly.plotly as py

from plotly.graph_objs import *



trace1 = {

  "x": ["Global", "USA", "Great Britain", "Mexico", "Taiwan", "Singapore"], 

  "y": [7.31, 3.61, 13.1, 3.3, 5.6, 8.41], 

  "name": "Ed Sheeran", 

  "type": "bar", 

  "uid": "955263", 

  "xsrc": "sweetmusicality:5:824c9c", 

  "ysrc": "sweetmusicality:5:ef572e"

}

trace2 = {

  "x": ["Global", "USA", "Great Britain", "Mexico", "Taiwan", "Singapore"], 

  "y": [3.91, 0, 2.85, 0, 6.15, 6.26], 

  "name": "The Chainsmokers", 

  "type": "bar", 

  "uid": "067b27", 

  "xsrc": "sweetmusicality:5:824c9c", 

  "ysrc": "sweetmusicality:5:c677a0"

}

trace3 = {

  "x": ["Global", "USA", "Great Britain", "Mexico", "Taiwan", "Singapore"], 

  "y": [4.28, 6.92, 5.64, 0, 0, 0], 

  "name": "Drake", 

  "type": "bar", 

  "uid": "41897b", 

  "xsrc": "sweetmusicality:5:824c9c", 

  "ysrc": "sweetmusicality:5:ddcce6"

}

trace4 = {

  "x": ["Global", "USA", "Great Britain", "Mexico", "Taiwan", "Singapore"], 

  "y": [0, 0, 0, 3.8, 0, 0], 

  "name": "J Balvin", 

  "type": "bar", 

  "uid": "1e2640", 

  "xsrc": "sweetmusicality:5:824c9c", 

  "ysrc": "sweetmusicality:5:0e6f78"

}

trace5 = {

  "x": ["Global", "USA", "Great Britain", "Mexico", "Taiwan", "Singapore"], 

  "y": [0, 0, 0, 3.56, 0, 0], 

  "name": "Maluma", 

  "type": "bar", 

  "uid": "82752c", 

  "xsrc": "sweetmusicality:5:824c9c", 

  "ysrc": "sweetmusicality:5:5279b7"

}

trace6 = {

  "x": ["Global", "USA", "Great Britain", "Mexico", "Taiwan", "Singapore"], 

  "y": [0, 0, 0, 0, 0, 2.43], 

  "name": "Bruno Mars", 

  "type": "bar", 

  "uid": "7d1dea", 

  "xsrc": "sweetmusicality:5:824c9c", 

  "ysrc": "sweetmusicality:5:91136f"

}

trace7 = {

  "x": ["Global", "USA", "Great Britain", "Mexico", "Taiwan", "Singapore"], 

  "y": [0, 5.53, 0, 0, 0, 0], 

  "name": "Kendrick Lamar", 

  "type": "bar", 

  "uid": "e3f565", 

  "xsrc": "sweetmusicality:5:824c9c", 

  "ysrc": "sweetmusicality:5:7a8be3"

}

trace8 = {

  "x": ["Global", "USA", "Great Britain", "Mexico", "Taiwan", "Singapore"], 

  "y": [0, 0, 0, 0, 2.09, 0], 

  "name": "Martin Garrix", 

  "type": "bar", 

  "uid": "c384a5", 

  "xsrc": "sweetmusicality:5:824c9c", 

  "ysrc": "sweetmusicality:5:554b1c"

}



# create the data and layout of the graph

data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8]

layout = {

  "annotations": [

    {

      "x": 1.08896492729, 

      "y": 0.659509202454, 

      "font": {"size": 21}, 

      "showarrow": False, 

      "text": "<b>Artist</b>", 

      "xanchor": "middle", 

      "xref": "paper", 

      "yanchor": "bottom", 

      "yref": "paper"

    }

  ], 

  "autosize": True, 

  "barmode": "stack", 

  "font": {"size": 18}, 

  "hovermode": "closest", 

  "legend": {

    "x": 1.01935845381, 

    "y": 0.673239347844, 

    "borderwidth": 0, 

    "orientation": "v", 

    "traceorder": "normal"

  }, 

  "title": "<b>Top 3 Streamed Artists on Spotify from Jan 2017 - Aug 2017 by Country</b>", 

  "titlefont": {"size": 28}, 

  "xaxis": {

    "autorange": True, 

    "range": [-0.5, 5.5], 

    "title": "<b>Country</b>", 

    "type": "category"

  }, 

  "yaxis": {

    "autorange": True, 

    "range": [0, 22.7263157895], 

    "title": "<b>% this artist was streamed in its country</b>", 

    "type": "linear"

  }

}



# let's plot it!

fig = go.Figure(data = data, layout = layout)

py.iplot(fig)