import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline

%config InlineBackend.figure_format = 'svg'



import seaborn as sns

sns.set()
top_songs = pd.read_csv("../input/top2018.csv")
top_songs.describe()
top_songs.info()
top_songs["name_len"] = top_songs["name"].apply(lambda x: len(x))

top_songs["name_len"].plot(kind="hist")

plt.title("Length of the song name in characters")
def get_additional_info_from_songname(songname, info_needed = "songname"):

    """

    Accepts:

        songname (str). String from dataframe.

        info_needed (str). Either "songname" (cleaned from <feat.> and <with>) or 

                           "coauthor" (in colab with whom this song was recorded) or

                           "wordcount" (how many words the title contains)

    Splits it.

    Finds <(feat.> and <(with> parts.

    Gets rid of them.

    Returns either the length of a song name without this additional info or the info itself.

    """

    song_components = songname.split()

    

    if "(feat." in song_components:

        if info_needed == "songname":

            return " ".join(song_components[ : song_components.index("(feat.")])

        if info_needed == "coauthor":

            coauth = song_components[song_components.index("(feat.")+1:]

            # this is to get rid of the last parenthesis

            coauth[-1] = coauth[-1][:-1]

            return " ".join(coauth)

        if info_needed == "wordcount":

            return len(song_components[ : song_components.index("(feat.")])

    elif "(with" in song_components:

        if info_needed == "songname":

            return " ".join(song_components[ : song_components.index("(with")])

        if info_needed == "coauthor":

            coauth = song_components[song_components.index("(with")+1:]

            # this is to get rid of the last parenthesis

            coauth[-1] = coauth[-1][:-1]

            return " ".join(coauth)

        if info_needed == "wordcount":

            return len(song_components[ : song_components.index("(with")])

    else:

        # if there was no feat. or with in the song title

        if info_needed == "songname":

            return songname

        elif info_needed == "coauthor":

            return "No colab"

        elif info_needed == "wordcount":

            return len(songname.split())
top_songs["clean_name"] = top_songs["name"].apply(

                            lambda x: get_additional_info_from_songname(x, 

                                                                        info_needed="songname"))



top_songs["name_wordcount"] = top_songs["name"].apply(

                                lambda x: get_additional_info_from_songname(x,

                                                                            info_needed="wordcount"))



top_songs["collaborator"] = top_songs["name"].apply(

                                lambda x: get_additional_info_from_songname(x,

                                                                            info_needed="coauthor"))
top_songs["name_wordcount"].value_counts().sort_values(ascending=False).plot(kind="bar")

plt.title("Number of words in song name")

plt.xticks(rotation=0)
top_songs[top_songs["name_wordcount"] > 4]["clean_name"]
top_songs.at[21, "clean_name"] = "Te Bot?"



top_songs.at[46, "clean_name"] = "Finesse (Remix)"

top_songs.at[46, "collaborator"] = "Cardi B"



top_songs.at[47, "clean_name"] = "Back To You"



top_songs.at[84, "clean_name"] = "Perfect"

top_songs.at[84, "collaborator"] = "Beyonce"



top_songs.at[98, "clean_name"] = "Dusk Till Dawn"
top_songs["is_collab"] = top_songs["collaborator"].apply(lambda x: 0 if x == "No colab" else 1)



top_songs["is_collab"].value_counts().plot(kind="bar")

plt.title("Is the song a collaboration or not?")

plt.xticks([0,1],["No", "Yes"], rotation=0)
top_songs["artists"].value_counts()[top_songs["artists"].value_counts() >= 2]
top_songs["collaborator"].value_counts().head()
total_number_of_works = top_songs["artists"].value_counts()



for artist, number in top_songs["collaborator"].value_counts().iteritems():

    try:

        total_number_of_works.loc[artist] += number

    except:

        pass
total_number_of_works[total_number_of_works >= 2].sort_values(ascending=False)
plt.figure(figsize=(10, 3))

total_number_of_works[total_number_of_works >= 2].sort_values(ascending=False).plot(kind="bar")

plt.title("Number of songs per artist")

plt.xticks(rotation=81)
top_songs["danceability"].plot(kind='hist')

plt.title("Distribution of songs danceability")
top_artists = list(total_number_of_works[total_number_of_works >= 2].sort_values(ascending=False).index)
top_songs[top_songs["artists"].isin(top_artists)]["danceability"].plot(kind="hist")

plt.title("Danceability of most popular artists' songs")
top_songs["energy"].plot(kind='hist', bins=20)

plt.title("Energy Distribution")
fig = sns.jointplot(x="danceability", y="energy", data=top_songs)

# fig.fig.set_figwidth(10)

# fig.fig.set_figheight(5)
top_songs["is_by_top_artist"] = top_songs["artists"].apply(lambda x: 1 if x in top_artists else 0)
fig = sns.scatterplot(x="danceability", y="energy", data=top_songs, hue="is_by_top_artist")
top_songs["key"].value_counts().sort_index().plot(kind="bar")

plt.title("How many songs of given key are there?")

_ = plt.xticks(range(len(top_songs["key"].value_counts().sort_index().index)),

           ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],

           rotation=0)
top_songs["key_string"] = top_songs["key"].replace({0: "C", 1: "C#", 2: "D", 3: "D#", 

                                                    4: "E", 5: "F", 6: "F#", 7: "G",

                                                    8: "G#", 9: "A", 10: "A#", 11: "B"})
plt.figure(figsize=(8, 5))

sns.scatterplot(x="danceability", y="energy", data=top_songs, hue="key_string")

plt.title("Does the key has any influence on danceability / energy of the song?")
top_songs["loudness"].plot(kind="hist", bins=25)

plt.title("Loudness distribution")
g = sns.jointplot(x="loudness", y="energy", data=top_songs)

g.fig.suptitle("Loudness vs. Energy")
sns.lmplot(x="loudness", y="energy", data=top_songs)
sns.boxplot(x="key_string", y="loudness", data=top_songs)
top_songs["mode"].value_counts().plot(kind='bar', color=sns.color_palette("deep", 2))

_ = plt.xticks([0,1],["Major", "Minor"], rotation=0)
sns.scatterplot(x="danceability", 

                y="energy", 

                data=top_songs, 

                hue="mode", 

                palette=sns.color_palette("deep", 2)[::-1])
sns.scatterplot(x="key_string", y="loudness", data=top_songs, 

                hue="mode", palette=sns.color_palette("deep", 2)[::-1])
top_songs.groupby(["key_string","mode"]).size().unstack().plot(kind="bar", 

                                                               stacked=True, 

                                                               color = sns.color_palette("deep", 2)[::-1])

_ = plt.xticks(rotation=0)

_ = plt.title("Major / Minor Mode by Key")
top_songs["speechiness"].plot(kind='hist', bins=20)

_ = plt.title("Speechiness Distribution")
top_songs["acousticness"].plot(kind="hist", bins=20)

_ = plt.title("Acousticness Distribution")
sns.scatterplot(x="acousticness", y="loudness", 

                data=top_songs, hue="key_string",

                palette=sns.color_palette("magma", 12))

_ = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.scatterplot(x="acousticness", y="energy", 

                data=top_songs)

_ = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

_ = plt.title("Acousticness vs. Energy")
top_songs["liveness"].plot(kind="hist", bins=20)

_ = plt.title("Liveness distribution")
top_songs["valence"].plot(kind="hist", bins=30)

_ = plt.title("Valence Distribution")
sns.scatterplot(x="valence", y="danceability", data=top_songs)

_ = plt.title("Valence vs. Danceability")
sns.scatterplot(x="valence", y="energy", data=top_songs)

_ = plt.title("Valence vs. Energy")
top_songs["tempo"].plot(kind="hist", bins=20)

_ = plt.title("Tempo Distribution")
fig, ax = plt.subplots(1, 2, figsize=(10, 3))

sns.scatterplot(x="tempo", y="danceability", data=top_songs, ax=ax[0])

sns.scatterplot(x="tempo", y="energy", data=top_songs, ax=ax[1])
sns.boxplot(x="key_string", y="tempo", data=top_songs)

_ = plt.title("Keys vs. Tempo")
top_songs["duration_ms"].plot(kind="hist", bins=30)

_ = plt.title("Duration Distribution")
sns.boxplot(x="key_string", y="duration_ms", data=top_songs)