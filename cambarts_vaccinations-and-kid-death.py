# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib # for pandas plotting style

import seaborn as sns

from mpl_toolkits import mplot3d

matplotlib.style.use('ggplot')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
indicators = pd.read_csv("../input/Indicators.csv")

indicators
# How many Unique Indicators are there? This is a huge csv

indicators["IndicatorName"].nunique()
# Making a list of indicators, and putting indicators that we want into buckets

all_indicators = list(indicators.IndicatorName.unique())

deaths = [x for x in all_indicators if "death" in x.lower()]

vaccines = [x for x in all_indicators if "immunization" in x.lower()]

pop = [x for x in all_indicators if "population" in x.lower()]

pop
#Found an indicator involving numbe of children under five

kid_death = indicators["IndicatorName"] == "Number of under-five deaths"

kid_death_df = indicators[kid_death]

kid_death_df["id"] = kid_death_df.CountryCode + kid_death_df.Year.astype(str)

kid_death_df.set_index("id", inplace=True)

kid_death_df.drop(columns=["IndicatorName", "IndicatorCode", "Year"], inplace=True)

kid_death_df
# Lets make similar DataFrames for the two vaccines

# DPT

dpt = indicators.IndicatorName == vaccines[0]

dpt_df = indicators[dpt]

dpt_df["id"] = dpt_df.CountryCode + dpt_df.Year.astype(str)

dpt_df.set_index("id", inplace=True)

dpt_df.drop(columns=["CountryName", "CountryCode","IndicatorName", "IndicatorCode", "CountryCode", "Year"], inplace=True)



# Measles

measles = indicators.IndicatorName == vaccines[1]

measles_df = indicators[measles]

measles_df["id"] = measles_df.CountryCode + measles_df.Year.astype(str)

measles_df.set_index("id", inplace=True)

measles_df.drop(columns=["CountryName", "CountryCode","IndicatorName", "IndicatorCode", "CountryCode", "Year"], inplace=True)

# And for total population

total_pop = indicators.IndicatorName == 'Population, total'

total_pop_df = indicators[total_pop]

total_pop_df["id"] = total_pop_df.CountryCode + total_pop_df.Year.astype(str)

total_pop_df.set_index("id", inplace=True)

total_pop_df.drop(columns=["CountryName", "CountryCode","IndicatorName", "IndicatorCode", "CountryCode"], inplace=True)

total_pop_df
# Okay lets attempt some sweet joins, so we have one good DF

vacc_df = dpt_df.join(measles_df, lsuffix="_dpt", rsuffix="_measles", on="id")

vacc_and_pop_df = vacc_df.join(total_pop_df, rsuffix="_population")

death_and_shots = vacc_and_pop_df.join(kid_death_df, rsuffix="_DeathUnderFive")

death_and_shots.dropna(inplace=True)

death_and_shots

death_and_shots.rename(index=str, columns={"Value_dpt": "DPT%", "Value_measles": "Measles%", "Value":"TotalPop", "Value_DeathUnderFive":"NumDeathUnderFive"}, inplace=True)
# Lets make a column for the number of deaths under five per 1000 people in country

death_and_shots["ChildDeathPer1000"] = (death_and_shots["NumDeathUnderFive"] / death_and_shots["TotalPop"]) * 1000
# Wait, you mean I might be able to finally make a chart? 

death_and_shots.plot.scatter(x="ChildDeathPer1000", y="DPT%")
# Well Color me shocked, if your kid doesn't get their shots, they are more likely to die



#Lets Add Some Dimension

death_and_shots.plot.scatter(x="ChildDeathPer1000", y="DPT%", c="Year",colormap='viridis')

np.corrcoef(death_and_shots["Measles%"], death_and_shots["ChildDeathPer1000"])
death_and_shots.corr()

fig = matplotlib.pyplot.figure()

ax = matplotlib.pyplot.axes(projection='3d')

ax.scatter(death_and_shots["ChildDeathPer1000"], death_and_shots["DPT%"],death_and_shots["Year"], c=death_and_shots["Year"], cmap='plasma')

# ax.view_init(40, 270)
# Lets try to get some region data



url = 'https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv'

regions_df = pd.read_csv(url)

regions_df.rename(columns={"alpha-3":"CountryCode"}, inplace=True)

regions_df = regions_df[["CountryCode", "sub-region"]]

regions_df

regions_i = regions_df.set_index("CountryCode")

death_and_shots_i = death_and_shots.set_index("CountryCode")

new_death_and_shots = death_and_shots_i.join(regions_i, on="CountryCode")

new_death_and_shots["sub-region"].nunique()
subs = list(new_death_and_shots["sub-region"].unique())

new_death_and_shots["sub-region-index"] = new_death_and_shots["sub-region"].apply(lambda x: subs.index(x))

new_death_and_shots.dropna()
fig = matplotlib.pyplot.figure()

ax = matplotlib.pyplot.axes(projection='3d')

ax.scatter(new_death_and_shots["ChildDeathPer1000"], new_death_and_shots["DPT%"],new_death_and_shots["Year"], c=new_death_and_shots["sub-region-index"], cmap='plasma')
# new_death_and_shots.plot.scatter(x="ChildDeathPer1000", y="DPT%", c="sub-region-index", colormap="viridis")

import matplotlib.pyplot as plt

sub_saharan = new_death_and_shots[new_death_and_shots["sub-region-index"]==5]

# matplotlib.pyplot.scatter(x=sub_saharan["ChildDeathPer1000"], y=sub_saharan["DPT%"])

plots = []

for year in range(1980, 2015):

    df = sub_saharan[sub_saharan["Year"]==year]

    plt.figure()

    plt.axis([0,18,0,100])

    plt.ylabel("% Of Population With DPT Vaccination")

    plt.xlabel("Deaths of Children Under 5 Per 1000 People")

    plt.title("Sub Saharan Africa " + str(year))

    plt.scatter(x=df["ChildDeathPer1000"], y=df["DPT%"])

    plt.savefig(f"{year}SubSaharan.jpeg")
import plotly.graph_objs as go

import plotly.plotly as py

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.io as pio



init_notebook_mode(connected=True)

go.Choropleth



def create_map(df, year):

    data = dict(type = 'choropleth',

               locations = df['CountryCode'],

                z = df['ChildDeathPer1000'],

                text = df['CountryName'],

                colorscale= "Reds",

                zmin=0,

                zmax=20,

                autocolorscale=False,

                colorbar= {'title':'# of Deaths'}

               )



    layout = dict(title = f'(Age < 5) Deaths Per 100 People ({year})',

                 geo=dict(showframe=False,

                         projection={"type":"natural earth"}))

    choromap = go.Figure(data=[data], layout=layout)

    iplot(choromap)



for year in range(1980, 2015):

    df = death_and_shots[death_and_shots["Year"]==year]

    create_map(df, year)