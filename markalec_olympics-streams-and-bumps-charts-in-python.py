import pandas as pd
summer = pd.read_csv("../input/summer.csv")
summer.head()
summer = summer[["Year", "Sport", "Country", "Gender", "Event", "Medal"]].drop_duplicates()
summer = summer.groupby(["Country", "Year"])["Medal"].count().unstack()
summer.head()
countries = [
    "USA", # United States of America
    "CHN", # China
    "RU1", "URS", "EUN", "RUS", # Russian Empire, USSR, Unified Team (post-Soviet collapse), Russia
    "GDR", "FRG", "EUA", "GER", # East Germany, West Germany, Unified Team of Germany, Germany
    "GBR", "AUS", "ANZ", # Australia, Australasia (includes New Zealand)
    "FRA", # France
    "ITA" # Italy
]

sm = summer.loc[countries]
sm.loc["Rest of world"] = summer.loc[summer.index.difference(countries)].sum()
sm = sm[::-1]
country_colors = {
    "USA":"steelblue",
    "CHN":"sandybrown",
    "RU1":"lightcoral", "URS":"indianred", "EUN":"indianred", "RUS":"lightcoral",
    "GDR":"yellowgreen", "FRG":"y",  "EUA":"y", "GER":"y", 
    "GBR":"silver",
    "AUS":"darkorchid", "ANZ":"darkorchid",
    "FRA":"silver",
    "ITA":"silver",
    "Rest of world": "gainsboro"}
%matplotlib inline

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("ticks")
sns.set_context("notebook", font_scale=1.2)
colors = [country_colors[c] for c in sm.index]

plt.figure(figsize=(12,8))
sm.T.plot.bar(stacked=True, color=colors, ax=plt.gca())

# Reverse the order of labels, so they match the data
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1])

# Set labels and remove superfluous plot elements
plt.ylabel("Number of medals")
plt.title("Stacked barchart of select countries' medals at the Summer Olympics")
sns.despine()
sm[1916] = np.nan # WW1
sm[1940] = np.nan # WW2
sm[1944] = np.nan # WW2
sm = sm[sm.columns.sort_values()]
plt.figure(figsize=(12,8))
sm.T.plot.area(color=colors, ax=plt.gca(), alpha=0.5)

# Reverse the order of labels, so they match the data
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1])

# Set labels and remove superfluous plot elements
plt.ylabel("Number of medals")
plt.title("Stacked areachart of select countries' medals at the Summer Olympics")
plt.xticks(sm.columns, rotation=90)
sns.despine()
for bl in ["zero", "sym", "wiggle", "weighted_wiggle"]:
    plt.figure(figsize=(6, 4))
    f = plt.stackplot(sm.columns, sm.fillna(0), colors=colors, baseline=bl, alpha=0.5, linewidth=1)
    [a.set_edgecolor(sns.dark_palette(colors[i])[-2]) for i,a in enumerate(f)] # Edges to be slighter darker
    plt.title("Baseline: {}".format(bl))
    plt.axis('off')
    plt.show()
from scipy import interpolate

def streamgraph(dataframe, **kwargs):
    """ Wrapper around stackplot to make a streamgraph """
    X = dataframe.columns
    Xs = np.linspace(dataframe.columns[0], dataframe.columns[-1], num=1024)
    Ys = [interpolate.PchipInterpolator(X, y)(Xs) for y in dataframe.values]
    return plt.stackplot(Xs, Ys, labels=dataframe.index, **kwargs)
plt.figure(figsize=(12, 8))

# Add in extra rows that are zero to make the zero region sharper
sm[1914] = np.nan
sm[1918] = np.nan
sm[1938] = np.nan
sm[1946] = np.nan
sm = sm[sm.columns.sort_values()]

f = streamgraph(sm.fillna(0), colors=colors, baseline="wiggle", alpha=0.5, linewidth=1)
[a.set_edgecolor(sns.dark_palette(colors[i])[-2]) for i,a in enumerate(f)] # Edges to be slighter darker

plt.axis('off')

# Reverse the order of labels, so they match the data
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], loc=(1, 0.35))

# Instead of ticks, left-align the dates and names of the Summer Olympics
# Use annotate to draw a line to the edge. 
cities = pd.read_csv("../input/summer.csv")[["Year", "City"]].drop_duplicates().set_index("Year")["City"].to_dict()
for c in cities:
    plt.annotate(xy=(c, plt.ylim()[1]), 
                 xytext=(c, plt.ylim()[0]-100), s="{} {}".format(c, cities[c]), 
                 rotation=90, verticalalignment="bottom", horizontalalignment="center", alpha=0.5, zorder=1,
                 arrowprops={"arrowstyle":"-", "zorder":0, "color":"k", "alpha":0.5})

# Block out regions when the World Wars occurred
plt.axvspan(xmin=1914, xmax=1918, color='white')
plt.axvspan(xmin=1938, xmax=1946, color='white')

plt.show()
winter = pd.read_csv("../input/winter.csv")
winter = winter[["Year", "Sport", "Country", "Gender", "Event", "Medal"]].drop_duplicates()
winter = winter.groupby(["Country", "Year"])["Medal"].count().unstack()
from collections import defaultdict

def add_widths(x, y, width=0.1):
    """ Adds flat parts to widths """
    new_x = []
    new_y = []
    for i,j in zip(x,y):
        new_x += [i-width, i, i+width]
        new_y += [j, j, j]
    return new_x, new_y

def bumpsplot(dataframe, color_dict=defaultdict(lambda: "k"), 
                         linewidth_dict=defaultdict(lambda: 1),
                         labels=[]):
    r = dataframe.rank(method="first")
    r = (r - r.max() + r.max().max()).fillna(0) # Sets NAs to 0 in rank
    for i in r.index:
        x = np.arange(r.shape[1])
        y = r.loc[i].values
        color = color_dict[i]
        lw = linewidth_dict[i]
        x, y = add_widths(x, y, width=0.1)
        xs = np.linspace(0, x[-1], num=1024)
        plt.plot(xs, interpolate.PchipInterpolator(x, y)(xs), color=color, linewidth=lw, alpha=0.5)
        if i in labels:
            plt.text(x[0] - 0.1, y[0], s=i, horizontalalignment="right", verticalalignment="center", color=color)
            plt.text(x[-1] + 0.1, y[-1], s=i, horizontalalignment="left", verticalalignment="center", color=color)
    plt.xticks(np.arange(r.shape[1]), dataframe.columns)
winter_colors = defaultdict(lambda: "grey")
lw = defaultdict(lambda: 1)

top_countries = winter.iloc[:, 0].dropna().sort_values().index
for i,c in enumerate(top_countries):
    winter_colors[c] = sns.color_palette("husl", n_colors=len(top_countries))[i]
    lw[c] = 4

plt.figure(figsize=(18,12))
bumpsplot(winter, color_dict=winter_colors, linewidth_dict=lw, labels=top_countries)
plt.gca().get_yaxis().set_visible(False)

cities = pd.read_csv("../input/winter.csv")[["Year", "City"]].drop_duplicates().set_index("Year")["City"]
plt.xticks(np.arange(winter.shape[1]), ["{} - {}".format(c, cities[c]) for c in cities.index], rotation=90)

# Add in annotation for particular countries
host_countries = {
    1924: "FRA",
    1928: "SUI",
    1932: "USA",
    1948: "SUI",
    1952: "NOR",
    1960: "USA",
    1964: "AUT",
    1968: "FRA",
    1976: "AUT",
    1980: "USA",
    1988: "CAN",
    1992: "FRA",
    1994: "NOR",
    2002: "USA",
    2010: "CAN",
}
for i,d in enumerate(winter.columns):
    if d in host_countries:
        plt.axvspan(i-0.1, i+0.1, color=winter_colors[host_countries[d]], zorder=0, alpha=0.5)

sns.despine(left=True)
x = [0, 1, 2, 3, 4]
xs = np.linspace(0, 4)
y = [0, 1, 1, 1, 0]

plt.plot(x, y, 'o-k', label="raw", alpha=0.5, markersize=16)
plt.plot(xs, interpolate.interp1d(x, y, kind="cubic")(xs), label="cubic spline", lw=4, alpha=0.8)
plt.plot(xs, interpolate.pchip(x, y)(xs), label="PCHIP", lw=4, alpha=0.8)

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
sns.despine()
