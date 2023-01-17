#NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
import numpy as np
#pandas is a software library written for the Python programming language for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series.
import pandas as pd


#Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy.
import matplotlib as mpl
import matplotlib.pyplot as plt # For base plotting
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
import seaborn as sns # Easier plotting


# Misc
import os
## To Show graphs in same window
%matplotlib inline
# Setting up Matplotlib, Seaborn map styles
mpl.style.use("seaborn")
plt.style.use("seaborn")

btui = [
    "#b2182b", "#d6604d", "#f4a582", "#92c5de", "#4393c3", "#2166ac", "#762a83",
    "#9970ab", "#c2a5cf", "#a6dba0", "#5aae61", "#1b7837", "#c51b7d", "#de77ae",
    "#f1b6da", "#8c510a", "#bf812d", "#dfc27d", "#80cdc1", "#35978f", "#01665e",
    ]
import random
btui_reversed = btui[::-1]
btui_shuffled=random.sample(btui, len(btui))

sns.set(context="notebook", style="darkgrid", font="monospace", font_scale=1.5, palette=btui)
sns.color_palette(btui)
sns.set_palette(btui)
sns.set(rc={"figure.figsize": (14, 10)})
#gunfile = 'gun-violence-data_01-2013_03-2018.csv'
#data_gun_violence = pd.read_csv(inputFolder+gunfile)

# Kaggle
# Read data file
data_gun_violence = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")
# Explore data - First 5 records of Gun Violance data
data_gun_violence.head()                          # head()
data_gun_violence.columns
data_gun_violence.columns.values
data_gun_violence.values
data_gun_violence.dtypes
data_gun_violence.describe()
data_gun_violence.info()
data_gun_violence.shape
# using isnull to find out missing values
data_gun_violence.isnull().values.any()

data_gun_violence.isnull().sum()

# using isna to find out missing values
data_gun_violence.isna().values.any()
sum_missing_data=data_gun_violence.isna().sum()
sum_missing_data
count_missing_data=data_gun_violence.isna().count()
count_missing_data
percentage_missing_data=(sum_missing_data/count_missing_data) * 100
percentage_missing_data
missing_data = pd.concat([sum_missing_data, percentage_missing_data], axis=1)
missing_data

del(sum_missing_data,count_missing_data,percentage_missing_data)
from matplotlib import collections as collections
from matplotlib.patches import Rectangle

#To install this package with conda run:
#conda install -c auto more-itertools 

#from itertools import izip as zip  #throwing error

import itertools
zip = getattr(itertools, 'izip', zip)
from itertools import cycle
# https://stackoverflow.com/questions/21925114/is-there-an-implementation-of-missingmaps-in-pythons-ecosystem
def missmap(df, ax=None, colors=None, aspect=4, sort='descending',
            title=None, **kwargs):
    """
    Plot the missing values of df.

    Parameters
    ----------
    df : pandas DataFrame
    ax : matplotlib axes
        if None then a new figure and axes will be created
    colors : dict
        dict with {True: c1, False: c2} where the values are
        matplotlib colors.
    aspect : int
        the width to height ratio for each rectangle.
    sort : one of {'descending', 'ascending', None}
    title : str
    kwargs : dict
        matplotlib.axes.bar kwargs

    Returns
    -------
    ax : matplotlib axes

    """
    if ax is None:
        fig, ax = plt.subplots()

    # setup the axes
    dfn = pd.isnull(df)

    if sort in ('ascending', 'descending'):
        counts = dfn.sum()
        sort_dict = {'ascending': True, 'descending': False}
        counts.sort_values(ascending=sort_dict[sort])
        dfn = dfn[counts.index]

    ny = len(df)
    nx = len(df.columns)
    # each column is a stacked bar made up of ny patches.
    xgrid = np.tile(np.arange(nx), (ny, 1)).T
    ygrid = np.tile(np.arange(ny), (nx, 1))
    # xys is the lower left corner of each patch
    xys = (zip(x, y) for x, y in zip(xgrid, ygrid))

    if colors is None:
        colors = {True: '#EAF205', False: 'k'}

    widths = cycle([aspect])
    heights = cycle([1])

    for xy, width, height, col in zip(xys, widths, heights, dfn.columns):
        color_array = dfn[col].map(colors)

        rects = [Rectangle(xyc, width, height, **kwargs)
                 for xyc, c in zip(xy, color_array)]

        p_coll = collections.PatchCollection(rects, color=color_array,
                                             edgecolor=color_array, **kwargs)
        ax.add_collection(p_coll, autolim=False)

    # post plot aesthetics
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)

    ax.set_xticks(.5 + np.arange(nx))  # center the ticks
    ax.set_xticklabels(dfn.columns)
    for t in ax.get_xticklabels():
        t.set_rotation(90)

    # remove tick lines
    ax.tick_params(axis='both', which='both', bottom='off', left='off',
                   labelleft='off')
    ax.grid(False)

    if title:
        ax.set_title(title)
    return ax

colours = {True: "#FF7256", False: "#ADD8E6"}
ax = missmap(data_gun_violence, colors = colours)
plt.show(ax)
data_gun_violence.drop([
    "incident_url",
    "sources",
    "source_url",
    "incident_url_fields_missing",
    "location_description",
    "participant_relationship",
    ], axis=1, inplace=True)
#Converting object datatype to datetime
data_gun_violence["date"] = pd.to_datetime(data_gun_violence["date"])
data_gun_violence["day"] = data_gun_violence["date"].dt.day
data_gun_violence["month"] = data_gun_violence["date"].dt.month
data_gun_violence["year"] = data_gun_violence["date"].dt.year
data_gun_violence["weekday"] = data_gun_violence["date"].dt.weekday
data_gun_violence["week"] = data_gun_violence["date"].dt.week
data_gun_violence["quarter"] = data_gun_violence["date"].dt.quarter
#Check the datatype of columns now
data_gun_violence.dtypes
data_gun_violence["gun_type"].unique()              #which values
# Created a new column for the total number of persons impacted (injured+killed) as per the data available
data_gun_violence["total_impacted"] = data_gun_violence["n_killed"] + data_gun_violence["n_injured"]
# Creating multiple columns from Participant's Gender column
data_gun_violence["participant_gender"] = data_gun_violence["participant_gender"].fillna("0::Unknown")


def clean_participant_gender(row) :
    gender_row_values = []
    gender_row = str(row).split("||")
    for x in gender_row :
        gender_row_value = str(x).split("::")
        if len(gender_row_value) > 1 :
            gender_row_values.append(gender_row_value[1])
            
    return gender_row_values


participant_genders = data_gun_violence.participant_gender.apply(clean_participant_gender)
data_gun_violence["participant_gender_total"] = participant_genders.apply(lambda x: len(x))
data_gun_violence["participant_gender_male"] = participant_genders.apply(lambda x: x.count("Male"))
data_gun_violence["participant_gender_female"] = participant_genders.apply(lambda x: x.count("Female"))
data_gun_violence["participant_gender_unknown"] = participant_genders.apply(lambda x: x.count("Unknown"))
del(participant_genders)
# Checking for null value of column for guns involved and guns stolen and filling the missing values
data_gun_violence["n_guns_involved"] = data_gun_violence["n_guns_involved"].fillna(value =0)
data_gun_violence["gun_stolen"] = data_gun_violence["gun_stolen"].fillna(value = "0::Unknown")
# Prints a lot but gives all the unique values of a column
#data_gun_violence["gun_stolen"].unique()

def clean_gun_stolen(row) :
    unknownCount = 0
    stolenCount = 0
    notstolenCount = 0
    gunstolen_row_values = []
    
    gunstolen_row = str(row).split("||")
    for x in gunstolen_row :
            gunstolen_row_value = str(x).split("::")
            if len(gunstolen_row_value) > 1 :
                gunstolen_row_values.append(gunstolen_row_value[1])
                if "Stolen" in gunstolen_row_value :
                    stolenCount += 1
                elif "Not-stolen" in gunstolen_row_value :
                    notstolenCount += 1
                else :
                    unknownCount += 1
                    
    return gunstolen_row_values


gunstolenvalues = data_gun_violence.gun_stolen.apply(clean_gun_stolen)
data_gun_violence["gun_stolen_stolen"] = gunstolenvalues.apply(lambda x: x.count("Stolen"))
data_gun_violence["gun_stolen_notstolen"] = gunstolenvalues.apply(lambda x: x.count("Not-stolen"))
del(gunstolenvalues)
# Checking values for new columns added
data_gun_violence.head()
# Checking the dimensions
data_gun_violence.shape
# Jointplot between Number of Person Killed Vs Injured in all incidences
sns.jointplot("n_injured",
              "n_killed",
              data_gun_violence,
              kind='scatter',      # kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }, optional
              s=200, color='m', edgecolor="skyblue", linewidth=2)
# Jointplot to identify Maximum Number of Person Injured in which incidence
sns.jointplot("incident_id",
              "n_injured",
              data_gun_violence,
              kind='scatter'      # kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }, optional
              )
# Jointplot to identify Maximum Number of Person Killed in which incidence
sns.jointplot("incident_id",
              "n_killed",
              data_gun_violence,
              kind='scatter',      # kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }, optional
              color="Red",
              marginal_kws={'color': 'red'})
# Jointplot to identify the number guns involved and the number of guns stolen
sns.jointplot(x=data_gun_violence["n_guns_involved"], y=data_gun_violence["gun_stolen_stolen"], kind="scatter", color="#D81B60")
# Jointplot to identify the number guns involved and the number of guns not stolen
sns.jointplot(x=data_gun_violence["n_guns_involved"], y=data_gun_violence["gun_stolen_notstolen"], kind="scatter", color="#1E88E5")
# Jointplot to identify the number of people killed based on gender wise participant total
sns.jointplot(x=data_gun_violence.participant_gender_total, y=data_gun_violence.n_killed, data=data_gun_violence, space=0, dropna=True, color="#D81B60")
# Jointplot to identify the number of people injured based on gender wise participant total
sns.jointplot(x=data_gun_violence.participant_gender_total, y=data_gun_violence.n_injured, data=data_gun_violence, space=0, dropna=True, color="#1E88E5")
# Jointplot to see the number of guns involved along with the number of people killed
sns.jointplot(x=data_gun_violence.n_guns_involved, y=data_gun_violence.n_killed, data=data_gun_violence, space=0, dropna=True, color="#D81B60")
# Histogram for Top 10 Cities with maximum incidents of Gun Violence
citywise_total = data_gun_violence[["incident_id"]].groupby(data_gun_violence["city_or_county"]).count()
top_cities = citywise_total.sort_values(by='incident_id', ascending=False).head(10)
print(top_cities)
top_cities.plot.barh()
del(top_cities)
# Histogram for Top 10 States with maximum incidents of Gun Violence
statewise_total = data_gun_violence[["incident_id"]].groupby(data_gun_violence["state"]).count()
top_states = statewise_total.sort_values(by='incident_id', ascending=False).head(10)
print(top_states)
top_states.plot.barh()
del(top_states)
# Histogram for Weekday wise Incidents
weekwise_total = data_gun_violence[["incident_id"]].groupby(data_gun_violence["weekday"]).count()
weekwise_total.plot.barh()
del(weekwise_total)
# Here, for weekdays 0 is for Monday and 6 is for Sunday.
# Histogram showing the crime rate state wise
state_vs_crimecount=sns.countplot(x=data_gun_violence["state"],data=data_gun_violence,order=data_gun_violence["state"].value_counts().index)
state_vs_crimecount.set_xticklabels(state_vs_crimecount.get_xticklabels(),rotation=90)
state_vs_crimecount.set_title("State Vs Crime Rate")
# Histogram showing the the top 10 cities with high crime rate
city_vs_crimerate=data_gun_violence['city_or_county'].value_counts().head(10)
city_vs_crimerate=sns.barplot(x=city_vs_crimerate.index,y=city_vs_crimerate)
city_vs_crimerate.set_xticklabels(city_vs_crimerate.get_xticklabels(),rotation=45)
city_vs_crimerate.set_title("Top 10 Cities having high crime rate")
# unique states
data_gun_violence['state'].unique()
# state wise crime count for top 10 states
state_vs_crimecount=data_gun_violence['state'].value_counts().head(10)
state_vs_crimecount
# Pie chart showing Top 10 States having high crime rate
plt.pie(state_vs_crimecount,labels=state_vs_crimecount.index,shadow=True)
plt.title("Top 10 States having high crime rate")
plt.axis("equal")
# the same graph in another way
plt.pie(state_vs_crimecount, labels=state_vs_crimecount.index, colors=btui, autopct="%1.1f%%", shadow=True, startangle=195)
plt.title("Top 10 States having high crime rate")
plt.axis("equal")
# state wise crime rate for all the states
statewise_crime_rate = data_gun_violence["state"].value_counts()
statewise_crime_rate
# Pie chart showing state wise crime rate for all the states
plt.pie(statewise_crime_rate, labels=statewise_crime_rate.index, colors=btui, autopct="%1.1f%%", shadow=True, startangle=195)
plt.title("State-wise Gun Violence Percentage")
plt.axis("equal")
# Top 50 cities having highest crime rate
topcitywise_crime_rate = data_gun_violence["city_or_county"].value_counts().head(50)
topcitywise_crime_rate
# Pie chart showing Top 50 cities having highest crime rate
plt.pie(topcitywise_crime_rate, labels=topcitywise_crime_rate.index, colors=btui, autopct="%1.1f%%", shadow=True, startangle=195)
plt.title("City-wise Gun Violence Percentage")
plt.axis("equal")
# Density plot for genderwise participant
genderwise_total = data_gun_violence[["participant_gender_total", "participant_gender_male", "participant_gender_female", "participant_gender_unknown"]].groupby(data_gun_violence["year"]).sum()
dp_gen_plot=sns.kdeplot(genderwise_total["participant_gender_male"], shade=True, color="r")
dp_gen_plot=sns.kdeplot(genderwise_total["participant_gender_female"], shade=True, color="b")
dp_gen_plot=sns.kdeplot(genderwise_total['participant_gender_unknown'], shade=True, color="g")
del(genderwise_total)
# Density plot for person injured vs killed on all weekdays
inj_kill_weektotal = data_gun_violence[["n_injured","n_killed"]].groupby(data_gun_violence["weekday"]).sum()
dp_inj_kill_plot=sns.kdeplot(inj_kill_weektotal['n_injured'], shade=True, color="r")
dp_inj_kill_plot=sns.kdeplot(inj_kill_weektotal['n_killed'], shade=True, color="b")
del(inj_kill_weektotal)
# Point plot showing yearly no of persons Killed 
yearly_vs_killed=data_gun_violence.groupby(data_gun_violence["year"]).apply(lambda x: pd.Series(dict(No_Killed=x.n_killed.sum())))
yearly_vs_killed_plot=sns.pointplot(x=yearly_vs_killed.index, y=yearly_vs_killed.No_Killed, data=yearly_vs_killed,label="yearly_vs_killed")
yearly_vs_killed
# Point plot showing yearly no of persons Injured
yearly_vs_injured=data_gun_violence.groupby(data_gun_violence["year"]).apply(lambda x: pd.Series(dict(No_Injured=x.n_injured.sum())))
yearly_vs_injured_plot=sns.pointplot(x=yearly_vs_injured.index, y=yearly_vs_injured.No_Injured, data=yearly_vs_injured,label="yearly_vs_injured")
yearly_vs_injured
# Point plot showing monthly no of people Killed 
monthly_vs_killed=data_gun_violence.groupby(data_gun_violence["month"]).apply(lambda x: pd.Series(dict(No_Killed=x.n_killed.sum())))
monthly_vs_killed_plot=sns.pointplot(x=monthly_vs_killed.index, y=monthly_vs_killed.No_Killed, data=monthly_vs_killed,label="monthly_vs_killed")
monthly_vs_killed
# Point plot showing monthly no of people injured
monthly_vs_injured=data_gun_violence.groupby(data_gun_violence["month"]).apply(lambda x: pd.Series(dict(No_Injured=x.n_injured.sum())))
monthly_vs_injured_plot=sns.pointplot(x=monthly_vs_injured.index, y=monthly_vs_injured.No_Injured, data=monthly_vs_injured,label="monthly_vs_injured")
monthly_vs_injured
# Violin Plot for Year wise no of people injured
year_vs_injured_plot = sns.violinplot("year", "n_injured", data=data_gun_violence,split=True, inner="quartile")
year_vs_injured_plot.set_title("Persons injured in the incidents per Year")
# Violin Plot for Year wise no of people killed
year_vs_killed_plot = sns.violinplot("year", "n_killed",
               data=data_gun_violence,
               split=True,         # If hue variable has two levels, draw half of a violin for each level.
               inner="quartile"    #  Options: “box”, “quartile”, “point”, “stick”, None 
               )
year_vs_killed_plot.set_title("Persons killed in the incidents per Year")
# Violin Plot for total num of persons Impacted(Killed/Injured) during gun violence
Impacted_persons_total = data_gun_violence[["total_impacted", "n_injured", "n_killed"]].groupby(data_gun_violence["year"]).sum()
print(Impacted_persons_total)

year_impacted_plot = sns.violinplot(data=Impacted_persons_total,
               split=True,         # If hue variable has two levels, draw half of a violin for each level.
               inner="quartile"    #  Options: “box”, “quartile”, “point”, “stick”, None 
               )
year_impacted_plot.set_title("Total number of persons Impacted(Killed/Injured) during gun violence")
del(Impacted_persons_total)
# Violin Plot for Genderwise total number of persons involved/impacted during gun violence
genderwise_total = data_gun_violence[["participant_gender_total", "participant_gender_male", "participant_gender_female", "participant_gender_unknown"]].groupby(data_gun_violence["year"]).sum()
print(genderwise_total)

year_genderwise_plot = sns.violinplot(data=genderwise_total,
               split=True,         # If hue variable has two levels, draw half of a violin for each level.
               inner="quartile"    #  Options: “box”, “quartile”, “point”, “stick”, None 
               )
year_genderwise_plot.set_title("Genderwise total number of persons involved/impacted during gun violence")
del(genderwise_total)
# State Vs No of People Killed
statewise_vs_killed=data_gun_violence.groupby(data_gun_violence["state"]).apply(lambda x: pd.Series(dict(No_Killed=x.n_killed.sum())))
statewise_vs_killed
# Box plot for total number of persons killed State wise
sns.boxplot('state','n_killed',data=data_gun_violence)
#State Vs No of people Injured
statewise_vs_injured=data_gun_violence.groupby(data_gun_violence["state"]).apply(lambda x: pd.Series(dict(No_Injured=x.n_injured.sum())))
statewise_vs_injured
# Box plot for total number of persons injured State wise
sns.boxplot('state','n_injured',data=data_gun_violence)
# Box Plot for Monthwise total number of Persons Killed
monthwise_killed_plot = sns.boxplot("month", "n_killed", data= data_gun_violence)
monthwise_killed_plot.set_title("Person killed in incidents per month")
# Box Plot for Monthwise total number of Persons Injured
month_injured_plot = sns.boxplot("month", "n_injured", data= data_gun_violence)
month_injured_plot.set_title("Person injured in incidents per month")
# Count Plot for Statewise incidences of Gun Violence
statewise_inc_plot = sns.countplot(x=data_gun_violence["state"], data = data_gun_violence,palette=btui,order=data_gun_violence["state"].value_counts().index)
statewise_inc_plot.set_title("Statewise incidence of Gun Violence")
statewise_inc_plot.set_xticklabels(statewise_inc_plot.get_xticklabels(), rotation=90)
# Count plot for statewise crime rate 
statewise_crime_rate = sns.countplot(x=data_gun_violence["state"], data=data_gun_violence, palette=btui, order=data_gun_violence["state"].value_counts().index)
statewise_crime_rate.set_xticklabels(statewise_crime_rate.get_xticklabels(), rotation=90)
statewise_crime_rate.set_title("State(s) with highest number of Gun Violence")
# Count Plot for State House District wise
state_housewise_inc_plot = sns.countplot("state_house_district", data = data_gun_violence,palette=btui,order=data_gun_violence["state_house_district"].value_counts().index)
state_housewise_inc_plot.set_title("State House District wise incidence of Gun Violence")
state_housewise_inc_plot.set_xticklabels(state_housewise_inc_plot.get_xticklabels(),rotation=90)
# Count Plot for State Senate District wise
state_incident_plot = sns.countplot("state_senate_district", data = data_gun_violence,palette=btui,order=data_gun_violence["state_house_district"].value_counts().index)
state_incident_plot.set_title("State Senate District wise incidence of Gun Violence")
state_incident_plot.set_xticklabels(state_incident_plot.get_xticklabels(),rotation=90)
# Facet Grid Graph for Male/ Female Partipant per Year
g = sns.FacetGrid(data_gun_violence, hue="year", palette="Set1", size=5, hue_kws={"marker": ["^", "v","*",">","<","+"]})
g.map(plt.scatter, "participant_gender_male", "participant_gender_female", s=100, linewidth=.5, edgecolor="white")
g.add_legend();
# Facet Grid Graphh for Person killed and Injured per Year
g = sns.FacetGrid(data_gun_violence, hue="year", palette="Set1", size=5, hue_kws={"marker": ["^", "v","*",">","<","o"]})
g.map(plt.scatter, "n_injured", "n_killed", s=100, linewidth=.5, edgecolor="white")
g.add_legend();
# Facet Grid Graphh for Person killed and Injured on Particular days of the week
g = sns.FacetGrid(data_gun_violence, hue="weekday", palette="Set1", size=5, hue_kws={"marker": ["^", "v","h","o","+",">","d"]})
g.map(plt.scatter, "n_injured", "n_killed", s=100, linewidth=.5, edgecolor="white")
g.add_legend();