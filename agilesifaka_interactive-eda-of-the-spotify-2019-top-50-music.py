import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
top_musics = pd.read_csv("../input/top50spotify2019/top50.csv",encoding="ISO-8859-1")

top_musics.head()
# not include the 1st colmun now

top_musics = top_musics[top_musics.columns[1:]]

top_musics.head()
# standardize the names of the column to increase readibility and accessibility

top_musics.columns = ['Track Name', 'Artist Name', 'Genre', 'Beats per Minute', 'Energy',

       'Danceability', 'Loudness (dB)', 'Liveness', 'Valence', 'Length',

       'Acousticness', 'Speechiness', 'Popularity']

top_musics.head()
# my personal reusable function for detecting missing data

def missing_value_describe(data):

    # check missing values in the data

    missing_value_stats = (data.isnull().sum() / len(data)*100)

    missing_value_col_count = sum(missing_value_stats > 0)

    missing_value_stats = missing_value_stats.sort_values(ascending=False)[:missing_value_col_count]

    print("Number of columns with missing values:", missing_value_col_count)

    if missing_value_col_count != 0:

        # print out column names with missing value percentage

        print("\nMissing percentage (desceding):")

        print(missing_value_stats)

    else:

        print("No missing data!!!")

missing_value_describe(top_musics)
# sort the table based on the popularity

top_musics = top_musics.sort_values(by=['Popularity'])

top_musics.head()
artist_contribution = top_musics[["Artist Name","Genre", "Popularity"]]



import plotly.express as px

fig = px.box(top_musics, x="Artist Name", y="Popularity")

fig.update_layout(

    title_text = "Top 50 Tracks Popularity Range by Artist (Hover Chart Markers to Interact)",

    xaxis=dict(

        tickangle=45

    )

)

fig.show()
fig = px.scatter(top_musics, x="Artist Name", y="Popularity", color="Genre", hover_name="Track Name")

fig.update_layout(

    title_text = "Top 50 Tracks Popularity by Artist and Genre (Hover to See the Song Name)",

    xaxis=dict(tickangle=45)

)

print("The legend on the right is also clickable. Try single clicks and also double clicks!")

fig.show()
fig = px.box(top_musics, x="Genre", y="Popularity")

fig.update_layout(

    title_text = "Top 50 Tracks' Genre Popularity (Hover Chart Markers to Interact)",

    xaxis=dict(tickangle=45)

)

fig.show()
# group by artist and take the average in one single line

top_musics_mean = top_musics.groupby("Artist Name").mean()

top_musics_mean.reset_index(level=0, inplace=True) # set index as column

top_musics_mean = top_musics_mean.sort_values(by=['Popularity']) # sort rows by popularity

top_musics_mean.head()

top_musics_mean.head()
len(top_musics_mean["Artist Name"])
# let's take a look at the popularity first

top_musics_mean["is top 10"] = [0] * (-10 + len(top_musics_mean["Artist Name"])) + [1] * 10

fig = px.bar(top_musics_mean, x="Artist Name", y="Popularity", color="is top 10")

fig.update_layout(

    title_text = "Mean Track Popularity by Artist (Hover Chart Markers to Interact)",

    xaxis=dict(tickangle=45)

)

fig.show()
top_musics.iloc[:,3:].describe()
correlation=top_musics.iloc[:,3:].corr(method='pearson')

correlation
import plotly.graph_objects as go

fig = go.Figure(

    data = [

        go.Heatmap(

            z=correlation, x=correlation.columns, y=correlation.index,

            hoverongaps = False

        )

    ],

    layout = go.Layout(

        title_text = "Correlations of the numeric scores of the Spotify top 50 tracks",

        autosize = True,

        width = 650,

        height = 650

    )

)

fig.show()
# calculate the number of tracks by genre

Genre_counts = top_musics["Genre"].value_counts()

Genre_counts_index = Genre_counts.index

Genre_counts, Genre_counts_index = zip(*sorted(zip(Genre_counts, Genre_counts_index)))



fig = go.Figure(

    data = go.Bar( x=Genre_counts_index, y=Genre_counts),

    layout = go.Layout(

        title_text = "Number of Tracks by Genre of the Spotify Top 50 Music List",

        yaxis = dict(title_text="Track Count"),

        xaxis = dict(title_text="Track Genre")

    )

)

fig.show()
# treemap for visualizing proportions

fig = go.Figure(

    go.Treemap(

        labels = ["Number of Tracks by Genre of the Spotify Top 50 Music List"] + list(Genre_counts_index),

        parents = [""] + ["Number of Tracks by Genre of the Spotify Top 50 Music List"] * len(Genre_counts_index),

        values = [0] + list(Genre_counts),

        textposition='middle center', # center the text

        textinfo = "label+percent parent", # show label and its percentage among the whole treemap

        textfont=dict(

            size=15 # adjust small text to larger text

        )

    )  

)

fig.show()



# radar chart for visualization the proportion

fig = go.Figure(

    go.Scatterpolar(

        r=Genre_counts,

        theta=Genre_counts_index,

        fill='toself'

    )

)

fig.update_layout(

    title_text = "Proportion of track by genre",

    title_x=0.5,

    polar=dict(radialaxis=dict(visible=True),),

    showlegend=False

)



fig.show()
# sort 2 lists together

track_names = top_musics["Track Name"]

track_names = [name.split("(")[0].strip() for name in track_names] # remove the feature part of the names

track_names = [name.split("-")[0].strip() for name in track_names] # remove the feature part of the names

track_names = list(set(track_names))

for i in range(len(track_names)):

    if "Sunflower" in track_names[i]:

        track_names[i] = "Sunflower"

track_name_length = [len(track.split(" ")) for track in track_names]

track_name_length, track_names = zip(*sorted(zip(track_name_length,track_names)))



# bar plot

fig = go.Figure(

    data = go.Bar(x=track_names, y=track_name_length),

    layout = go.Layout(

        title_text = "Track Name Lengths of the Spotify Top 50 Music List",

        yaxis = dict(title_text="Name Length by Word"),

        xaxis = dict(title_text="Track Name"),

        autosize = True,

        height=600

    )

)

fig.show()



# create labels for donut chart

labels = []

for i in range(len(track_name_length)):

    if track_name_length[i] == 1:

        labels.append("1 word name")

    else:

        labels.append(str(track_name_length[i]) + " words name")

fig = go.Figure(

    data=[go.Pie(labels=labels, values=[1] * len(track_name_length), hole=.3)],

    layout = go.Layout(

        title_text = "Proportions of the Track Name Lengths of the Spotify Top 50 Music List",

    ))

fig.show()
# sort 2 lists together

artist_names = top_musics["Artist Name"]

artist_names = [name.split("(")[0].strip() for name in artist_names] # remove the feature part of the names

artist_names = [name.split("-")[0].strip() for name in artist_names] # remove the feature part of the names

artist_names = list(set(artist_names))

artist_names_length = [len(track.split(" ")) for track in artist_names]

artist_names_length, artist_names = zip(*sorted(zip(artist_names_length,artist_names)))



# bar plot

fig = go.Figure(

    data = go.Bar(x=artist_names, y=artist_names_length),

    layout = go.Layout(

        title_text = "Artist Name Lengths of the Spotify Top 50 Music List",

        yaxis = dict(title_text="Name Length by Word"),

        xaxis = dict(title_text="Artist Name"),

        autosize = True,

        height=600

    )

)

fig.show()





# create labels for donut chart

labels = []

for i in range(len(artist_names_length)):

    if artist_names_length[i] == 1:

        labels.append("1 word name")

    else:

        labels.append(str(artist_names_length[i]) + " words name")

fig = go.Figure(

    data=[go.Pie(labels=labels, values=[1] * len(artist_names_length), hole=.3)],

    layout = go.Layout(

        title_text = "Proportions of the Artist Name Lengths of the Spotify Top 50 Music List",

    ))

fig.show()