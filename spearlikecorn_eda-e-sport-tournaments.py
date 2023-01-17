import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ommit one of the warnings
import warnings
warnings.filterwarnings("ignore", message="Glyph 146 missing from current font.")
pd.options.mode.chained_assignment = None

# read the data
esport = pd.read_csv("/kaggle/input/esports-earnings/EsportsEarnings_final.csv", encoding = "ISO-8859-1")

esport
esport.info()
esport.describe()
esport[esport['ReleaseDate']==esport['ReleaseDate'].min()]
# checking if there are any entries with Relase date smaller then 1900 which should be imposible
esport[esport.ReleaseDate<1900]
esport.loc[1576, 'ReleaseDate'] = 2011
esport.describe()
esport.isnull().sum()
esport[["Year", "Month", "Day"]] = esport.Date.str.split('-', expand=True)
esport[["Year", "Month", "Day"]] = esport[["Year", "Month", "Day"]].astype(int)
# Data grouping and aggregation
esport_counts = esport[["Game", "Tournaments", "ReleaseDate", "Earnings", "Genre"]].groupby("Game").agg({"Tournaments": np.sum, "ReleaseDate": np.mean, 
                                                                                                        "Earnings": np.sum, "Genre": lambda x:x.value_counts().index[0]})
esport_counts.sort_values("Tournaments", ascending=False, inplace=True)
esport_counts.reset_index(inplace=True)

# create figure
fig, ax = plt.subplots(figsize=(30,15))
ax = sns.scatterplot(x = "ReleaseDate", y = "Tournaments", size = "Earnings", hue="Genre", data=esport_counts, sizes=(100, 10000), alpha=.5)
# find N games with the most numbers of tournaments
most_tournamets = esport_counts.nlargest(15, "Tournaments")
# add annotation to found games
for line in range(0,esport_counts.shape[0]):
    if esport_counts.Game[line] in list(most_tournamets.Game):
        ax.text(esport_counts.ReleaseDate[line], esport_counts.Tournaments[line], esport_counts.Game[line], horizontalalignment='center', 
                size='x-large', color='black', weight='semibold')
# modify legend
handles, labels = ax.get_legend_handles_labels()
to_skip = len(np.unique(esport_counts.Genre))+2
for h in handles[to_skip:]:
    sizes = [s / 100 for s in h.get_sizes()] # smaller Earnings scatter points on legend
    label = h.get_label()
    label = str(float(label)*100) +" mln"
    h.set_sizes(sizes) # set them
    h.set_label(label)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large', title_fontsize='40') # bigger legend font size
plt.legend(loc=2, fontsize='x-large')
ax.xaxis.label.set_size(20)
ax.xaxis.set_tick_params(labelsize='x-large')
ax.yaxis.label.set_size(20)
ax.yaxis.set_tick_params(labelsize='x-large')
plt.rc('axes', labelsize="x-large")    # fontsize of the axes labels
plt.title("Game release date to number of overall tournaments with overall prize pools", fontsize=24)
plt.show()
import calendar
# Data grouping and aggregation
esport_dota2 = esport[esport["Game"] == "Dota 2"]
esport_dota2 = esport_dota2.groupby(["Year", "Month"]).agg({"Tournaments": np.sum, "ReleaseDate": np.mean, "Earnings": np.mean, "Genre": lambda x:x.value_counts().index[0]})
esport_dota2.sort_values("Tournaments", ascending=False, inplace=True)
esport_dota2.reset_index(inplace=True)
esport_dota2["Month name"] = esport_dota2["Month"].apply(lambda x: calendar.month_abbr[x])
#esport_dota2["Date"] = pd.to_datetime(esport_dota2[["Year", "Month"]])]
plt.figure(figsize=(25,10))
plt.subplot(1,2,1)
plt.title("Dota 2 amount of tournaments", fontsize=24)
ax = sns.boxplot(x="Month name", y="Tournaments", data=esport_dota2)
ax.xaxis.label.set_size(20)
ax.xaxis.set_tick_params(labelsize='x-large')
ax.yaxis.label.set_size(20)
ax.yaxis.set_tick_params(labelsize='x-large')
plt.subplot(1,2,2)
plt.title("Dota 2 value of the rewards", fontsize=24)
ax = sns.boxplot(x="Month name", y="Earnings", data=esport_dota2)
ax.xaxis.label.set_size(20)
ax.xaxis.set_tick_params(labelsize='x-large')
ax.yaxis.label.set_size(20)
ax.yaxis.set_tick_params(labelsize='x-large')
plt.show()
# Data grouping and aggregation
esport_sc2 = esport[esport["Game"] == "StarCraft II"]
esport_sc2 = esport_sc2.groupby(["Year", "Month"]).agg({"Tournaments": np.sum, "ReleaseDate": np.mean, "Earnings": np.mean, "Genre": lambda x:x.value_counts().index[0]})
esport_sc2.sort_values("Tournaments", ascending=False, inplace=True)
esport_sc2.reset_index(inplace=True)
esport_sc2["Month name"] = esport_sc2["Month"].apply(lambda x: calendar.month_abbr[x])
#esport_dota2["Date"] = pd.to_datetime(esport_dota2[["Year", "Month"]])]
plt.figure(figsize=(25,10))
plt.subplot(1,2,1)
plt.title("StarCraft 2 amount of tournaments")
ax = sns.boxplot(x="Month name", y="Tournaments", data=esport_sc2)
ax.xaxis.label.set_size(20)
ax.xaxis.set_tick_params(labelsize='x-large')
ax.yaxis.label.set_size(20)
ax.yaxis.set_tick_params(labelsize='x-large')
plt.subplot(1,2,2)
plt.title("StarCraft 2 value of the rewards")
ax = sns.boxplot(x="Month name", y="Earnings", data=esport_sc2)
ax.xaxis.label.set_size(20)
ax.xaxis.set_tick_params(labelsize='x-large')
ax.yaxis.label.set_size(20)
ax.yaxis.set_tick_params(labelsize='x-large')
plt.show()
esport_NF = esport.copy()
# removing numbers, brackets, : and , signs
esport_NF.Game = esport_NF.Game.str.split('\s[+0-9()\']').str[0]
esport_NF.Game = esport_NF.Game.str.split('[:]').str[0]
# removing greek letters
esport_NF.replace("\sIII|\sII|\sIV|\sIX|\sVI|\sV|\sXIII|\sXII|\sXI|\sX|\sI|",'',regex=True, inplace=True)

# Data grouping and aggregation
esport_counts = esport_NF[["Game", "Tournaments", "ReleaseDate", "Earnings", "Genre"]].groupby("Game").agg({"Tournaments": np.sum, "ReleaseDate": np.mean, 
                                                                                                        "Earnings": np.sum, "Genre": lambda x:x.value_counts().index[0]})
esport_counts.sort_values("Tournaments", ascending=False, inplace=True)
esport_counts.reset_index(inplace=True)

esport_counts = esport_counts.nlargest(30, 'Tournaments')
fig, ax = plt.subplots(figsize=(20,20))
sns.barplot(esport_counts.Tournaments, esport_counts.Game)
ax.xaxis.tick_top()
ax.xaxis.label.set_size(20)
ax.xaxis.set_tick_params(labelsize='x-large')
ax.yaxis.label.set_size(20)
ax.yaxis.set_tick_params(labelsize='x-large')
plt.show()
fig, axes = plt.subplots(3, 3, figsize=(40,16))
names_MRG = game_money.Game[0:9]
names_MRG = iter(names_MRG)
for row in axes:
    for col in row:
        game_name = next(names_MRG)
        game_esport = esport[esport["Game"] == game_name]
        game_esport = game_esport.groupby("Year").mean()
        game_esport.reset_index(inplace=True)
        game_esport = game_esport[game_esport.Year != 2020] # 2020 has not ended
        col.scatter(game_esport.Year, game_esport.Earnings)
        col.plot(game_esport.Year, game_esport.Earnings)
        col.set_title(game_name, size=26)
esport_domination = esport.groupby("Genre").agg({"Tournaments": np.sum})
esport_domination.reset_index(inplace=True)
plt.figure(figsize=(26,6))
ax = sns.barplot(x="Genre", y="Tournaments", data = esport_domination)
ax.xaxis.label.set_size(20)
ax.xaxis.set_tick_params(rotation=90)
ax.xaxis.set_tick_params(labelsize='x-large')
ax.yaxis.label.set_size(20)
ax.yaxis.set_tick_params(labelsize='x-large')
plt.show()
# filter fighting games only and find its share in Tournaments
esport_fight = esport[esport["Genre"] == "Fighting Game"]
esport_fight = esport_fight[["Game", "Tournaments"]].groupby(by = "Game").sum()
esport_fight["Share"] = esport_fight["Tournaments"] / esport_fight["Tournaments"].sum()
# def figure
plt.figure(figsize=(20,5))
# define cmap
cmap = plt.get_cmap("Greens")
cmap = iter(cmap([i/10 for i in range(10)]))
# find biggest prized games - give them color
cols = []
labs = []
mostvalue = esport_fight.nlargest(10, 'Tournaments')
for i, gamename in enumerate(esport_fight.index):
    if gamename in mostvalue.index:
        cols.append(next(cmap))
        labs.append(gamename)
    else:
        cols.append("gray")
        labs.append("")
patches, texts = plt.pie(esport_fight.Tournaments, labels=labs, colors=cols, radius=2, textprops={'fontsize': 12})
plt.show()
esport_fight_NF = esport_fight.copy()
esport_fight_NF.reset_index(inplace=True)
# removing numbers, brackets, : and , signs
esport_fight_NF.Game = esport_fight_NF.Game.str.split('\s[+0-9()\']').str[0]
esport_fight_NF.Game = esport_fight_NF.Game.str.split('[:]').str[0]
esport_fight_NF.Game = esport_fight_NF.Game.str.split('\sXX').str[0]
# removing greek letters
esport_fight_NF.replace("\sIII|\sII|\sIV|\sIX|\sVI|\sV|\sXIII|\sXII|\sXI|\sXX|\sXrd|\sX|\sI|",'',regex=True, inplace=True)
#esport_fight_NF = esport_fight_NF.Game.str.findall("Street\sFighter|Soul\sCalibur|Super\sSmash\sBros|Guilty\sGear")
expresion = r"Street\sFighter|Soul\sCalibur|Super\sSmash\sBros|Tekken|Dragon\sBall|Dead\sor\sAlive|Guilty\sGear"
esport_fight_NF["Game"][esport_fight_NF.Game.str.contains(expresion)] = esport_fight_NF.Game.str.findall(expresion).str[0]
esport_fight_NF = esport_fight_NF.groupby("Game").sum()

plt.figure(figsize=(20,5))
# define cmap
cmap = plt.get_cmap("Greens")
cmap = iter(cmap([i/10 for i in range(10)]))
# find biggest prized games - give them color
cols = []
labs = []
mostvalue = esport_fight_NF.nlargest(10, 'Tournaments')
for i, gamename in enumerate(esport_fight_NF.index):
    if gamename in mostvalue.index:
        cols.append(next(cmap))
        labs.append(gamename)
    else:
        cols.append("gray")
        labs.append("")
patches, texts = plt.pie(esport_fight_NF.Tournaments, labels=labs, colors=cols, radius=2, textprops={'fontsize': 12})
#plt.legend(patches, esport_fight_NF.index, loc="upper right", bbox_to_anchor=(1,1),bbox_transform=plt.gcf().transFigure)
plt.show()
# filter fighting games only and find its share Earnings
esport_fight = esport[esport["Genre"] == "Fighting Game"]
esport_fight = esport_fight[["Game", "Earnings"]].groupby(by = "Game").mean()
esport_fight["Share"] = esport_fight["Earnings"] / esport_fight["Earnings"].sum()
# def figure
plt.figure(figsize=(20,5))
# define cmap
cmap = plt.get_cmap("Reds")
cmap = iter(cmap([i/10 for i in range(10)]))
# find biggest prized games - give them color
cols = []
labs = []
mostvalue = esport_fight.nlargest(10, 'Earnings')
for i, gamename in enumerate(esport_fight.index):
    if gamename in mostvalue.index:
        cols.append(next(cmap))
        labs.append(gamename)
    else:
        cols.append("gray")
        labs.append("")
patches, texts = plt.pie(esport_fight.Earnings, labels=labs, colors=cols, radius=2, textprops={'fontsize': 12})
#plt.legend(patches, esport_fight.index, loc="upper right", bbox_to_anchor=(1,1),bbox_transform=plt.gcf().transFigure)
plt.show()
esport_fight_NF = esport_fight.copy()
esport_fight_NF.reset_index(inplace=True)
# removing numbers, brackets, : and , signs
esport_fight_NF.Game = esport_fight_NF.Game.str.split('\s[+0-9()\']').str[0]
esport_fight_NF.Game = esport_fight_NF.Game.str.split('[:]').str[0]
esport_fight_NF.Game = esport_fight_NF.Game.str.split('\sXX').str[0]
# removing greek letters
esport_fight_NF.replace("\sIII|\sII|\sIV|\sIX|\sVI|\sV|\sXIII|\sXII|\sXI|\sXX|\sXrd|\sX|\sI|",'',regex=True, inplace=True)
#esport_fight_NF = esport_fight_NF.Game.str.findall("Street\sFighter|Soul\sCalibur|Super\sSmash\sBros|Guilty\sGear")
expresion = r"Street\sFighter|Soul\sCalibur|Super\sSmash\sBros|Tekken|Dragon\sBall|Dead\sor\sAlive|Guilty\sGear"
esport_fight_NF["Game"][esport_fight_NF.Game.str.contains(expresion)] = esport_fight_NF.Game.str.findall(expresion).str[0]
esport_fight_NF = esport_fight_NF.groupby("Game").sum()

plt.figure(figsize=(20,5))
# define cmap
cmap = plt.get_cmap("Reds")
cmap = iter(cmap([i/10 for i in range(10)]))
# find biggest prized games - give them color
cols = []
labs = []
mostvalue = esport_fight_NF.nlargest(10, 'Earnings')
for i, gamename in enumerate(esport_fight_NF.index):
    if gamename in mostvalue.index:
        cols.append(next(cmap))
        labs.append(gamename)
    else:
        cols.append("gray")
        labs.append("")
patches, texts = plt.pie(esport_fight_NF.Earnings, labels=labs, colors=cols, radius=2, textprops={'fontsize': 12})
#plt.legend(patches, esport_fight_NF.index, loc="upper right", bbox_to_anchor=(1,1),bbox_transform=plt.gcf().transFigure)
plt.show()