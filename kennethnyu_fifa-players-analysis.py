import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from pylab import rcParams



from plotly.offline import iplot, init_notebook_mode

from geopy.geocoders import Nominatim

import missingno as msno

import plotly.plotly as py



%matplotlib inline

rcParams["figure.figsize"] = 12, 8

rcParams["font.size"] = 3

rcParams["xtick.labelsize"] = 12

rcParams["ytick.labelsize"] = 12
data = pd.read_csv("../input/CompleteDataset.csv")
data.tail(3)
# Deleting useless columns

useless = ["Photo", "Flag", "Club Logo"]

for item in useless:

    try:

        del data[item]

    except:

        print("Already deleted " + item + " column.")
# Data availability

msno.matrix(data.sample(200))
# Plotting the age boxplot and distribution for players

fig, axes = plt.subplots(2)

sns.boxplot(x="Age", data=data, ax=axes[0])

sns.distplot(data["Age"], bins=len(data["Age"].value_counts().sort_index().index), ax=axes[1])

axes[1].set_xlim(min(data["Age"])-1, max(data["Age"]))

axes[0].set_xticks(range(min(data["Age"]), max(data["Age"])+1))

axes[1].set_xticks(range(min(data["Age"]), max(data["Age"])+1))

fig.tight_layout()
# Overall ratings

fig, axe = plt.subplots()

ax = sns.distplot(data["Overall"], bins=len(data["Overall"].value_counts().index))

ax.set_xticks(range(0, 100, 5))

ax.set_xlim(0,100)

ax.set_xlabel("Overall Rating")

fig.tight_layout()
print("Overall ratings ~ N({}, {})".format(

    round(data["Overall"].describe()[1], 1), 

    round(data["Overall"].describe()[2], 2)))
# Replacing England and Wales by UK

data["Country"] = data["Nationality"]

data["Country"] = data["Country"].replace(["England", "Wales"], "United Kingdom")

                                          

# Grouping the data by countries

valcon = data.groupby("Country").size().reset_index(name="Count")
# Plotting the choropleth map

init_notebook_mode()

plotmap = [ dict(

        type = 'choropleth',

        locations = valcon["Country"],

        locationmode = 'country names',

        z = valcon["Count"],

        text = valcon["Country"],

        autocolorscale = True,

        reversescale = False,

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) ),

        colorbar = dict(

            title = "Amount of Players"),

      ) ]



layout = dict(

    title = "Origins of FIFA 18 Players",

    geo = dict(

        showframe = False,

        showcoastlines = False,

        projection = dict(

            type = 'Mercator'

        )

    )

)



fig = dict( data=plotmap, layout=layout )

iplot(fig)
data["Country"].value_counts().head(25)