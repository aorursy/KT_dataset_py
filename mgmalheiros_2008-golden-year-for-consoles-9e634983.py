# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import seaborn as sns

sns.set_style("whitegrid")



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



data = pd.read_csv("../input/ign.csv")
data.head()
data["genre"].value_counts()
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



data["Type"] = data["platform"].map(mapping)
data["Type"].value_counts()
df = data.groupby(["Type", "release_year"]).size().unstack().T

df.reset_index(inplace = True)

df = df.fillna(0)
df.head()
df[["Arcade", "Computer", "Console", "Mobile", "Portable", "Other"]].plot(x = df["release_year"])