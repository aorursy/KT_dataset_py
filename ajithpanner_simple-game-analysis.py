import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
gamedata=pd.read_csv('../input/ign.csv')
gamedata.head()
#lets delete some messy column of array from this data frame

del gamedata['Unnamed: 0']

del gamedata['title']

del gamedata['url']

gamedata.head()
gamedata.describe()
gamedata["score_phrase"].value_counts()
gamedata.score_phrase.value_counts()[:6].plot.pie(figsize=(10,10))#top six scorephrase
positivescorephrase = ["Great", "Good", "Masterpiece", "Amazing"]

neither_positive_nor_negativescorephrase = ["Medicore", "Okay"]

negativescorephrase = ["Bad","Disaster","Awful","Painful","Unbearable"]

def mapping(item):

    if item in positivescorephrase:

        return "positivescorephrase"

    if item in neither_positive_nor_negativescorephrase:

        return "neither_positive_nor_negativescorephrase"

    if item in negativescorephrase:

        return "negativescorephrase"



gamedata["scorephrasetype"] = gamedata["score_phrase"].map(mapping)
gamedata["scorephrasetype"].value_counts()
gamedata.scorephrasetype.value_counts().plot.pie(figsize=(10,10))
gamedata["platform"].value_counts()
gamedata.platform.value_counts()[:6].plot.pie(figsize=(10,10))#top six platform
Computer = ["PC", "Macintosh", "Linux", "Commodore 64/128", "Windows Surface", "SteamOS"]

Console = ["PlayStation 2", "Xbox 360", "Wii", "PlayStation 3", "Nintendo DS", "PlayStation", "Xbox",

           "GameCube", "Nintendo 64", "Dreamcast", "PlayStation 4", "Xbox One", "Wii U", "Genesis",

           "NES", "TurboGrafx-16", "Super NES", "Sega 32X", "Master System", "Nintendo 64DD", "Saturn",

           "Atari 2600", "Atari 5200", "TurboGrafx-CD", "Ouya"]

Portable = ["Nintendo DSi", "PlayStation Portable", "Game Boy Advance", "Game Boy Color", "Nintendo 3DS",

            "PlayStation Vita", "Lynx", "NeoGeo Pocket Color", "Game Boy", "N-Gage", "WonderSwan",

            "New Nintendo 3DS", "WonderSwan Color", "dreamcast VMU"]

Mobile = ["iPhone", "iPad", "Android", "Windows Phone", "iPod", "Pocket PC"]

Arcade = ["Arcade", "NeoGeo", "Vectrex"]



def mapping(item):

    if item in Computer:

        return "Computer"

    if item in Console:

        return "Console"

    if item in Portable:

        return "Portable"

    if item in Mobile:

        return "Mobile"

    if item in Arcade:

        return "Arcade"

    return "Other"



gamedata["platformtype"] = gamedata["platform"].map(mapping)
gamedata["platformtype"].value_counts()
gamedata.platformtype.value_counts().plot.pie(figsize=(10,10))
scores=gamedata.score

scores.describe()
gamedata.score.value_counts()[:6.0].plot.pie(figsize=(10,10))#TOP firstclass SCORE
gamedata["genre"].value_counts()
gamedata.genre.value_counts()[:6].plot.pie(figsize=(10,10))#top six genre
gamedata["editors_choice"].value_counts()
gamedata.editors_choice.value_counts().plot.pie(figsize=(10,10))
gamedata["release_year"].value_counts()
gamedata.release_year.value_counts().plot.barh(figsize=(10,10))
gamedata.release_month. value_counts()
gamedata.release_month.value_counts().plot.barh(figsize=(10,10))
gamedata.release_day.value_counts()
gamedata.release_day.value_counts().plot. barh(figsize=(10,10))