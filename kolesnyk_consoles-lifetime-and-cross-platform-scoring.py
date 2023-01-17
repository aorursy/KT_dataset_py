# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib_venn import venn2

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



computer = ["PC", "Macintosh", "Linux", "Commodore 64/128", "Windows Surface", "SteamOS"]

con_ps = ["PlayStation", "PlayStation 2", "PlayStation 3", "PlayStation 4"]

con_xbox = ["Xbox", "Xbox 360", "Xbox One"]

con_nintendo = ["Wii","Wii U","Nintendo 64","Nintendo 64DD", "NES","Super NES","GameCube"]

con_sega = ["Sega 32X", "Dreamcast", "Genesis", "Saturn", "Master System"]

con_atari = ["Atari 2600", "Atari 5200"]

con_nec =["TurboGrafx-16", "TurboGrafx-CD"]

console = ["Ouya"] + con_nec + con_atari + con_sega + con_nintendo + con_xbox + con_ps

portable = ["Nintendo DS", "Nintendo DSi", "PlayStation Portable", "Game Boy Advance", "Game Boy Color", "Nintendo 3DS",

            "PlayStation Vita", "Lynx", "NeoGeo Pocket Color", "Game Boy", "N-Gage", "WonderSwan",

            "New Nintendo 3DS", "WonderSwan Color", "Dreamcast VMU"]

mobile = ["iPhone", "iPad", "Android", "Windows Phone", "iPod", "Pocket PC"]

arcade = ["Arcade", "NeoGeo", "Vectrex"]
raw_ign_df = pd.read_csv("../input/ign.csv")

print("Number of platforms: ", len(raw_ign_df["platform"].unique()))
sns.set_palette("husl")

plot = raw_ign_df.groupby("platform")["platform"].count().plot.bar()

plot.set_xlabel("")
not_rubbish = raw_ign_df.groupby("platform").count()

not_rubbish = not_rubbish[not_rubbish["title"] > 100]

raw_ign_df = raw_ign_df[(raw_ign_df["platform"].isin(not_rubbish.index)) & (raw_ign_df["release_year"] > 1995)]
plot = raw_ign_df.groupby("platform")["platform"].count().plot.bar()

plot.set_xlabel("")
platform = []

for item in console:

    if item in list(not_rubbish.index):

        platform.append(item)

        

data = raw_ign_df[raw_ign_df["platform"].isin(platform)]

df = data.groupby(["platform", "release_year"]).size().unstack().fillna(0).T

df.reset_index(inplace = True)



sns.set_palette("husl", len(platform))

ax = df[platform].plot(x=df["release_year"])

ax.set_ylabel("Titles, pcs.")

ax.legend(loc=9, ncol=4)
plt_lifetime = {}

for pl in platform: plt_lifetime[pl] = len(df[df[pl] > 0])

calculated_df = pd.DataFrame(pd.Series(plt_lifetime), columns=['lifetime'])

for pl in platform: calculated_df.loc[pl,"titles_av"] = round(df[pl].sum() / len(df[df[pl] > 0]),0)

calculated_df["lifetime"].plot.barh()

plot.set_xlabel("Lifetime")
titles_con = raw_ign_df.loc[raw_ign_df["platform"].isin(console),["title"]]["title"].unique(); print(titles_con, len(titles_con))

titles_com = raw_ign_df.loc[raw_ign_df["platform"].isin(computer),["title"]]["title"].unique(); print(titles_com, len(titles_com))

titles_con_com = set(titles_con).intersection(titles_com); print("Number of titles presented on consoles & computers:", len(titles_con_com))

titles_con_u = set(titles_con).difference(titles_com); print("Number of titles unique to consoles:", len(titles_con_u))

titles_com_u = set(titles_com).difference(titles_con); print("Number of titles unique to computers:", len(titles_com_u))

venn2(subsets=(len(titles_con), len(titles_com), len(titles_con_com)), set_labels = ('Console', 'Computer', 'Console & Computer'))
cross_rank_df = pd.DataFrame()

for title in titles_con_com:

    console_rnk = round(raw_ign_df[(raw_ign_df["platform"].isin(console)) & (raw_ign_df["title"] == title)]["score"].mean(), 1)

    computer_rnk = round(raw_ign_df[(raw_ign_df["platform"].isin(computer)) & (raw_ign_df["title"] == title)]["score"].mean(),1)

    cross_rank_df.loc[title, "console_rnk"] = console_rnk

    cross_rank_df.loc[title, "computer_rnk"] = computer_rnk

    cross_rank_df.loc[title, "ratio"] = computer_rnk/console_rnk



sns.jointplot(cross_rank_df["console_rnk"], cross_rank_df["computer_rnk"])