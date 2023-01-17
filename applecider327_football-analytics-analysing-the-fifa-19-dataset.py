import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import warnings

from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action = "ignore", category=SettingWithCopyWarning)

%matplotlib inline
fifa_19 = pd.read_csv("../input/fifa19/data.csv")
fifa_19.head(10)
fifa_19.columns
fifa_19 = fifa_19.drop(["Unnamed: 0", "ID", "Photo", "Flag", "Potential", "Club Logo", "Special", "Weak Foot", "Skill Moves", "Body Type", "Real Face", "Jersey Number", "Joined", "Loaned From", "Contract Valid Until", "LS", "ST", "RS", "LW", "LF", "CF", "RF", "RW",

"LAM", "CAM", "RAM", "LM", "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB", "CB", "RCB", "RB", "Release Clause"], axis = 1)
fifa_19.head(10)
epl = ["Manchester City", "Manchester United", "Tottenham Hotspur", "Liverpool", "Chelsea", "Arsenal", "Burnley", "Everton", "Leicester City", "Newcastle United", "Crystal Palace", "Bournemouth", "West Ham United", "Watford", "Brighton & Hove Albion", "Huddersfield Town", "Southampton", "Swansea City", "Stoke City", "West Bromwich Albion"]

la_liga = ["FC Barcelona", "Atlético Madrid", "Real Madrid", "Valencia CF", "Villarreal CF", "Real Betis", "Sevilla FC", "Getafe CF", "SD Eibar", "Girona FC", "RCD Espanyol", "Real Sociedad", "RC Celta", "Deportivo Alavés", "Levante UD", "Athletic Club de Bilbao", "CD Leganés", "Deportivo de La Coruña", "UD Las Palmas", "Málaga CF"]

bundesliga = ["FC Bayern München", "FC Schalke 04", "TSG 1899 Hoffenheim", "Borussia Dortmund", "Bayer 04 Leverkusen", "RB Leipzig", "VfB Stuttgart", "Eintracht Frankfurt", "Borussia Mönchengladbach", "Hertha BSC", "SV Werder Bremen", "FC Augsburg", "Hannover 96", "1. FSV Mainz 05", "SC Freiburg", "VfL Wolfsburg", "Hamburger SV", "1. FC Köln"]

serie_a = ["Juventus", "Napoli", "Roma", "Inter", "Lazio", "Milan", "Atalanta", "Fiorentina", "Torino", "Sampdoria", "Sassuolo", "Genoa", "Chievo Verona", "Udinese", "Bologna", "Cagliari", "SPAL", "Crotone", "Hellas Verona", "Benevento"]

ligue_1 = ["Paris Saint-Germain", "AS Monaco", "Olympique Lyonnais", "Olympique de Marseille", "Stade Rennais FC", "FC Girondins de Bordeaux", "AS Saint-Étienne", "OGC Nice", "FC Nantes", "Montpellier HSC", "Dijon FCO", "En Avant de Guingamp", "Amiens SC", "Angers SCO", "RC Strasbourg Alsace", "Stade Malherbe Caen", "LOSC Lille", "Toulouse Football Club", "ESTAC Troyes", "FC Metz"]

major_five = epl + la_liga + bundesliga + serie_a + ligue_1
def european_five(club):

    if club in major_five:

        return True

    else:

        return False
major_or_not = fifa_19["Club"].apply(european_five)

european_majors = fifa_19[major_or_not]

european_majors.set_index([pd.Index(range(european_majors.shape[0]))], inplace = True)
european_majors.head(20)
european_majors.isnull().sum()
missing = european_majors[european_majors["Position"].isnull()]

missing
european_majors.drop(labels = [2571, 2572, 2573, 2574], inplace = True)

european_majors.set_index([pd.Index(range(european_majors.shape[0]))], inplace = True)

european_majors.isnull().sum()
def EPL(club):

    if club in epl:

        return True

    else:

        return False
epl_or_not = european_majors["Club"].apply(EPL)

premier_league = european_majors[epl_or_not]

premier_league.set_index([pd.Index(range(premier_league.shape[0]))], inplace = True)

premier_league.head(20)
gk = premier_league[premier_league["Position"] == "GK"]

epl_gk = gk[:17]

epl_gk.drop(labels = [25, 38, 48, 96, 117, 123, 156], axis = 0, inplace = True)

epl_gk.set_index([pd.Index(range(epl_gk.shape[0]))], inplace = True)

epl_gk
epl_gk["CleanSheetRate(%)"] = pd.Series([49, 42, 44, 24, 26, 32, 31, 17, 26, 16], index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

epl_gk
plt.scatter(epl_gk["Overall"], epl_gk["CleanSheetRate(%)"])

plt.xlabel("Overall Rating")

plt.ylabel("Clean Sheet Rate(%)")

plt.title("How well do the overall ratings reflect reality? (Goalkeepers)")
epl_gk[["Overall", "CleanSheetRate(%)"]].corr()
defence = premier_league[premier_league["Position"].str.contains("[LCRW]B")]

epl_defence = defence[:43]

epl_defence.drop(labels = [21, 22, 30, 31, 32, 40, 43, 50, 55, 61, 65, 72, 78, 82, 83, 85, 86, 88, 91, 94, 98, 102, 105, 107, 111, 113, 114, 119, 121, 122, 126, 128, 129], axis = 0, inplace = True)

epl_defence.set_index([pd.Index(range(epl_defence.shape[0]))], inplace = True)

epl_defence
epl_defence["DFScore"] = pd.Series([18.6, 21.3, 12.9, 14.3, 18.6, 19.7, 12.9, 16.7, 19.3, 13.8], index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

epl_defence
plt.scatter(epl_defence["Overall"], epl_defence["DFScore"])

plt.xlabel("Overall Rating")

plt.ylabel("Defence Score")

plt.title("How well do the overall ratings reflect reality? (Defenders)")
epl_defence[["Overall", "DFScore"]].corr()
forward = premier_league[premier_league["Position"].str.contains("ST")]

epl_forward = forward[:16]

epl_forward.drop(labels = [45, 77, 127, 144, 153, 177], axis = 0, inplace = True)

epl_forward.set_index([pd.Index(range(epl_forward.shape[0]))], inplace = True)

epl_forward
epl_forward["AvgGoal"] = pd.Series([0.81, 0.84, 0.47, 0.44, 0.54, 0.35, 0.13, 0.29, 0.18, 0.19], index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

epl_forward
plt.scatter(epl_forward["Overall"], epl_forward["AvgGoal"])

plt.xlabel("Overall Rating")

plt.ylabel("Goals per Match")

plt.title("How well do overall ratings reflect reality? (Strikers)")
epl_forward[["Overall", "AvgGoal"]].corr()
european_majors.columns
defAvg = epl_defence[["Interceptions", "HeadingAccuracy", "Marking", "StandingTackle", "SlidingTackle"]].sum(axis = 1)

epl_defence["defAvg"] = defAvg / 5

epl_defence
STAvg = epl_forward[["Positioning", "Finishing", "ShotPower", "LongShots", "Volleys", "Penalties"]].sum(axis = 1)

epl_forward["STAvg"] = STAvg / 6

epl_forward
Avg = epl_gk[["GKDiving", "GKHandling", "GKKicking", "GKPositioning", "GKReflexes"]].sum(axis = 1)

epl_gk["GKavg"] = Avg / 5

epl_gk
plt.scatter(epl_gk["GKavg"], epl_gk["CleanSheetRate(%)"])

plt.xlabel("GK Attributes Average")

plt.ylabel("Clean Sheet Rate(%)")

plt.title("How well do the attribute ratings reflect reality? (Goalkeepers)")
epl_gk[["GKavg", "CleanSheetRate(%)"]].corr()
plt.scatter(epl_defence["defAvg"], epl_defence["DFScore"])

plt.xlabel("DF Attributes Average")

plt.ylabel("Defence Score")

plt.title("How well do the attribute ratings reflect reality? (Defenders)")
epl_defence[["defAvg", "DFScore"]].corr()
plt.scatter(epl_forward["STAvg"], epl_forward["AvgGoal"])

plt.xlabel("ST Attributes Average")

plt.ylabel("Goals per Match")

plt.title("How well do the attribute ratings reflect reality? (Strikers)")
epl_forward[["STAvg", "AvgGoal"]].corr()
def LaLiga(club):

    if club in la_liga:

        return True

    else:

        return False

    

laliga_or_not = european_majors["Club"].apply(LaLiga)

LaLiga = european_majors[laliga_or_not]

LaLiga.set_index([pd.Index(range(LaLiga.shape[0]))], inplace = True)

LaLiga.head(20)
def BundesLiga(club):

    if club in bundesliga:

        return True

    else:

        return False

    

bund_or_not = european_majors["Club"].apply(BundesLiga)

Bundesliga = european_majors[bund_or_not]

Bundesliga.set_index([pd.Index(range(Bundesliga.shape[0]))], inplace = True)

Bundesliga.head(20)
def SerieA(club):

    if club in serie_a:

        return True

    else:

        return False



serie_or_not = european_majors["Club"].apply(SerieA)

serieA = european_majors[serie_or_not]

serieA.set_index([pd.Index(range(serieA.shape[0]))], inplace = True)

serieA.head(20)
def Ligue1(club):

    if club in ligue_1:

        return True

    else:

        return False



ligue1_or_not = european_majors["Club"].apply(Ligue1)

Ligue1 = european_majors[ligue1_or_not]

Ligue1.set_index([pd.Index(range(Ligue1.shape[0]))], inplace = True)

Ligue1.head(20)
premier_league.head(20)
a = premier_league["Overall"].mean()

b = LaLiga["Overall"].mean()

c = Bundesliga["Overall"].mean()

d = serieA["Overall"].mean()

e = Ligue1["Overall"].mean()

added = [a] + [b] + [c] + [d] + [e]



figure = plt.figure(figsize = (18, 8))

ax = figure.add_subplot(1, 1, 1)

ax.bar([0.75, 1.75, 2.75, 3.75, 4.75], [a, b, c, d, e], 0.5, color = (173/255, 216/255, 230/255))

for each in range(5):

    ax.text(0.6 + each, 65, round(added[each], 2), fontsize = 20)

ax.set_xticks([0.75, 1.75, 2.75, 3.75, 4.75])

ax.set_xticklabels(["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"], fontsize = 20)

ax.set_yticklabels(range(0, 80, 10), fontsize = 18)

plt.title("Average Overall Player Rating across Major European Leagues, Season 2017-2018", fontsize = 24)
eplr = european_majors[epl_or_not]["Overall"]

laligar = european_majors[laliga_or_not]["Overall"]

bundesligar = european_majors[bund_or_not]["Overall"]

seriear = european_majors[serie_or_not]["Overall"]

liguer = european_majors[ligue1_or_not]["Overall"]



rfive = [eplr, laligar, bundesligar, seriear, liguer]

names = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]

fig, ax = plt.subplots(figsize = (12, 8))

for i in range(5):

    sns.kdeplot(rfive[i], shade = True, label = names[i])

plt.xlabel("Overall Rating", size = 16)

plt.ylabel("Relative Frequency", size = 16)

plt.show()
labels = european_majors[european_majors["Value"].str.contains("K")]["Value"].keys()

european_majors["Value"] = european_majors["Value"].str.replace("K", "")

european_majors["Value"] = european_majors["Value"].str.replace("M", "")

european_majors["Value"] = european_majors["Value"].str.replace("€", "")

european_majors["Value"] = european_majors["Value"].astype(float)

european_majors.iloc[labels, 5] = european_majors.iloc[labels, 5]/1000
eplmp = european_majors[epl_or_not]["Value"]

laligamp = european_majors[laliga_or_not]["Value"]

bundesligamp = european_majors[bund_or_not]["Value"]

serieamp = european_majors[serie_or_not]["Value"]

liguemp = european_majors[ligue1_or_not]["Value"]
mfive = [eplmp, laligamp, bundesligamp, serieamp, liguemp]

names = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]

fig, ax = plt.subplots(figsize = (12, 8))

for i in range(5):

    sns.kdeplot(mfive[i], shade = True, label = names[i])

plt.xlabel("Market Value, in millions of €", size = 16)

plt.ylabel("Relative Frequency", size = 16)

plt.show()
eplage = european_majors[epl_or_not]["Age"]

laligaage = european_majors[laliga_or_not]["Age"]

bundesligaage = european_majors[bund_or_not]["Age"]

serieaage = european_majors[serie_or_not]["Age"]

ligueage = european_majors[ligue1_or_not]["Age"]
fig, ax = plt.subplots(figsize = (12, 8))

ages = [eplage, laligaage, bundesligaage, serieaage, ligueage]

ax.boxplot(ages)

ax.set_ylabel("Age", size = 16)

ax.set_xticklabels(["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"], size = 14)

ax.set_title("How does player age distribution differ among the five major European leagues?", size = 16)

plt.show()
man_c = premier_league.loc[premier_league["Club"] == "Manchester City", "Overall"]

man_u = premier_league.loc[premier_league["Club"] == "Manchester United", "Overall"]

tot_h = premier_league.loc[premier_league["Club"] == "Tottenham Hotspur", "Overall"]

liv = premier_league.loc[premier_league["Club"] == "Liverpool", "Overall"]

barc = LaLiga.loc[LaLiga["Club"] == "FC Barcelona", "Overall"]

amad = LaLiga.loc[LaLiga["Club"] == "Atlético Madrid", "Overall"]

rmad = LaLiga.loc[LaLiga["Club"] == "Real Madrid", "Overall"]

val = LaLiga.loc[LaLiga["Club"] == "Valencia CF", "Overall"]

bayern = Bundesliga.loc[Bundesliga["Club"] == "FC Bayern München", "Overall"]

schalke = Bundesliga.loc[Bundesliga["Club"] == "FC Schalke 04", "Overall"]

hoff = Bundesliga.loc[Bundesliga["Club"] == "TSG 1899 Hoffenheim", "Overall"]

dortm = Bundesliga.loc[Bundesliga["Club"] == "Borussia Dortmund", "Overall"]

juv = serieA.loc[serieA["Club"] == "Juventus", "Overall"]

nap = serieA.loc[serieA["Club"] == "Napoli", "Overall"]

roma = serieA.loc[serieA["Club"] == "Roma", "Overall"]

inter = serieA.loc[serieA["Club"] == "Inter", "Overall"]

psg = Ligue1.loc[Ligue1["Club"] == "Paris Saint-Germain", "Overall"]

mon = Ligue1.loc[Ligue1["Club"] == "AS Monaco", "Overall"]

lyon = Ligue1.loc[Ligue1["Club"] == "Olympique Lyonnais", "Overall"]

mars = Ligue1.loc[Ligue1["Club"] == "Olympique de Marseille", "Overall"]
fig, ax = plt.subplots(figsize = (20, 16))

clubs = [man_c, man_u, tot_h, liv, barc, amad, rmad, val, bayern, schalke, hoff, dortm, juv, nap, roma, inter, psg, mon, lyon, mars]

box = ax.boxplot(clubs, patch_artist = True)

colors = []

for i in range(20):

    colors.append((173/255, 200/255, (50 + 10*i)/255))

for patch, color in zip(box['boxes'], colors):

    patch.set_facecolor(color)

ax.set_ylabel("Overall Rating", size = 20)

ax.set_xlabel("Clubs", size = 20)

ax.set_xticklabels(["Manchester City", "Manchester United", "Tottenham Hotspur", "Liverpool", "FC Barcelona", "Atlético Madrid", "Real Madrid", "Valencia CF", "FC Bayern München", "FC Schalke 04", "TSG 1899 Hoffenheim", "Borussia Dortmund", "Juventus", "Napoli", "Roma", "Inter", "Paris Saint-Germain", "AS Monaco", "Olympique Lyonnais", "Olympique de Marseille"], size = 14, rotation = 90)

ax.set_title("Overall Rating Distributions of Top 4 Clubs in their Respective Leagues", size = 20)

plt.show()
man_c2 = european_majors.loc[european_majors["Club"] == "Manchester City", "Value"]

man_u2 = european_majors.loc[european_majors["Club"] == "Manchester United", "Value"]

tot_h2 = european_majors.loc[european_majors["Club"] == "Tottenham Hotspur", "Value"]

liv2 = european_majors.loc[european_majors["Club"] == "Liverpool", "Value"]

barc2 = european_majors.loc[european_majors["Club"] == "FC Barcelona", "Value"]

amad2 = european_majors.loc[european_majors["Club"] == "Atlético Madrid", "Value"]

rmad2 = european_majors.loc[european_majors["Club"] == "Real Madrid", "Value"]

val2 = european_majors.loc[european_majors["Club"] == "Valencia CF", "Value"]

bayern2 = european_majors.loc[european_majors["Club"] == "FC Bayern München", "Value"]

schalke2 = european_majors.loc[european_majors["Club"] == "FC Schalke 04", "Value"]

hoff2 = european_majors.loc[european_majors["Club"] == "TSG 1899 Hoffenheim", "Value"]

dortm2 = european_majors.loc[european_majors["Club"] == "Borussia Dortmund", "Value"]

juv2 = european_majors.loc[european_majors["Club"] == "Juventus", "Value"]

nap2 = european_majors.loc[european_majors["Club"] == "Napoli", "Value"]

roma2 = european_majors.loc[european_majors["Club"] == "Roma", "Value"]

inter2 = european_majors.loc[european_majors["Club"] == "Inter", "Value"]

psg2 = european_majors.loc[european_majors["Club"] == "Paris Saint-Germain", "Value"]

mon2 = european_majors.loc[european_majors["Club"] == "AS Monaco", "Value"]

lyon2 = european_majors.loc[european_majors["Club"] == "Olympique Lyonnais", "Value"]

mars2 = european_majors.loc[european_majors["Club"] == "Olympique de Marseille", "Value"]
fig, ax = plt.subplots(figsize = (20, 16))

clubs = [man_c2, man_u2, tot_h2, liv2, barc2, amad2, rmad2, val2, bayern2, schalke2, hoff2, dortm2, juv2, nap2, roma2, inter2, psg2, mon2, lyon2, mars2]

sns.boxenplot(data = clubs, palette = 'rainbow')

ax.set_ylabel("Market Value, in millions of €", size = 20)

ax.set_xlabel("Clubs", size = 20)

ax.set_xticklabels(["Manchester City", "Manchester United", "Tottenham Hotspur", "Liverpool", "FC Barcelona", "Atlético Madrid", "Real Madrid", "Valencia CF", "FC Bayern München", "FC Schalke 04", "TSG 1899 Hoffenheim", "Borussia Dortmund", "Juventus", "Napoli", "Roma", "Inter", "Paris Saint-Germain", "AS Monaco", "Olympique Lyonnais", "Olympique de Marseille"], size = 14, rotation = 90)

ax.set_title("Market Value Distributions of Top 4 Clubs in their Respective Leagues", size = 20)

plt.show()
european_majors["Wage"] = european_majors["Wage"].str.replace("€", "").str.replace("K", "").astype("int")

man_c3 = european_majors.loc[european_majors["Club"] == "Manchester City", "Wage"]

man_u3 = european_majors.loc[european_majors["Club"] == "Manchester United", "Wage"]

tot_h3 = european_majors.loc[european_majors["Club"] == "Tottenham Hotspur", "Wage"]

liv3 = european_majors.loc[european_majors["Club"] == "Liverpool", "Wage"]

barc3 = european_majors.loc[european_majors["Club"] == "FC Barcelona", "Wage"]

amad3 = european_majors.loc[european_majors["Club"] == "Atlético Madrid", "Wage"]

rmad3 = european_majors.loc[european_majors["Club"] == "Real Madrid", "Wage"]

val3 = european_majors.loc[european_majors["Club"] == "Valencia CF", "Wage"]

bayern3 = european_majors.loc[european_majors["Club"] == "FC Bayern München", "Wage"]

schalke3 = european_majors.loc[european_majors["Club"] == "FC Schalke 04", "Wage"]

hoff3 = european_majors.loc[european_majors["Club"] == "TSG 1899 Hoffenheim", "Wage"]

dortm3 = european_majors.loc[european_majors["Club"] == "Borussia Dortmund", "Wage"]

juv3 = european_majors.loc[european_majors["Club"] == "Juventus", "Wage"]

nap3 = european_majors.loc[european_majors["Club"] == "Napoli", "Wage"]

roma3 = european_majors.loc[european_majors["Club"] == "Roma", "Wage"]

inter3 = european_majors.loc[european_majors["Club"] == "Inter", "Wage"]

psg3 = european_majors.loc[european_majors["Club"] == "Paris Saint-Germain", "Wage"]

mon3 = european_majors.loc[european_majors["Club"] == "AS Monaco", "Wage"]

lyon3 = european_majors.loc[european_majors["Club"] == "Olympique Lyonnais", "Wage"]

mars3 = european_majors.loc[european_majors["Club"] == "Olympique de Marseille", "Wage"]
fig, ax = plt.subplots(figsize = (20, 16))

clubs = [man_c3, man_u3, tot_h3, liv3, barc3, amad3, rmad3, val3, bayern3, schalke3, hoff3, dortm3, juv3, nap3, roma3, inter3, psg3, mon3, lyon3, mars3]

sns.boxenplot(data = clubs, palette = 'magma')

ax.set_ylabel("Weekly Wage, in thousands of €", size = 20)

ax.set_xlabel("Clubs", size = 20)

ax.set_xticklabels(["Manchester City", "Manchester United", "Tottenham Hotspur", "Liverpool", "FC Barcelona", "Atlético Madrid", "Real Madrid", "Valencia CF", "FC Bayern München", "FC Schalke 04", "TSG 1899 Hoffenheim", "Borussia Dortmund", "Juventus", "Napoli", "Roma", "Inter", "Paris Saint-Germain", "AS Monaco", "Olympique Lyonnais", "Olympique de Marseille"], size = 14, rotation = 90)

ax.set_title("Weekly Wage Distributions of Top 4 Clubs in their Respective Leagues", size = 20)

plt.show()
def UK(x):

    if x in ["England", "Scotland", "Wales", "Northern Ireland"]:

        return True

    else:

        return False

    

fifa_19.loc[fifa_19["Nationality"].apply(UK), "Nationality"] = "United Kingdom"

boole = fifa_19["Nationality"].value_counts() >= 20

nations = boole[boole].keys()



def selected(x):

    if x in nations:

        return True

    else:

        return False



fifa = fifa_19[fifa_19["Nationality"].apply(selected)]
countries = np.unique(fifa["Nationality"])

avgo = fifa.groupby("Nationality")["Overall"].agg(np.mean)

data = [dict(

        type = 'choropleth',

        locations = countries,

        z = avgo,

        locationmode = 'country names',

        text = "Overall",

        marker = dict(

            line = dict(color = 'rgb(0,0,0)', width = 1)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = 'Average Overall Rating')

            )

       ]



layout = dict(

    title = 'Average Overall Rating by Country',

    geo = dict(

        showframe = False,

        showocean = True,

        oceancolor = 'rgb(0,255,255)',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )



fig = dict(data=data, layout=layout)

py.iplot(fig, validate = False)
countries = np.unique(fifa["Nationality"])

avga = fifa.groupby("Nationality")["Age"].agg(np.mean)

data = [dict(

        type = 'choropleth',

        locations = countries,

        z = avga,

        locationmode = 'country names',

        text = "Overall",

        marker = dict(line = dict(color = 'rgb(0,0,0)', width = 1)),

        colorbar = dict(autotick = True, tickprefix = '', title = 'Average Age')

        )

       ]



layout = dict(

    title = 'Average Player Age by Country',

    geo = dict(

        showframe = False,

        showocean = True,

        oceancolor = 'rgb(0,255,255)',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

        ),

        lataxis = dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

        )

        ),

    )



figure = dict(data=data, layout=layout)

py.iplot(figure, validate = False)