# Data manipulation modules
import pandas as pd        # R-like data manipulation
import numpy as np         # n-dimensional arrays
import os
# For plotting
import matplotlib as mpl
import matplotlib.pyplot as plt      # For base plotting
# Seaborn is a library for making statistical graphics
# in Python. It is built on top of matplotlib and 
#  numpy and pandas data structures.
import seaborn as sns                # Easier plotting

## To Show graphs in same window
%matplotlib inline

mpl.style.use("seaborn")
plt.style.use("seaborn")
##Set working directory
print(os.listdir("../input"))
# Kaggle
# Read data file
gunvio_data = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")
gunvio_data.head()
### check Columns 
gunvio_data.columns
### check dtypes
gunvio_data.dtypes
### check values
gunvio_data.values
gunvio_data.describe()
gunvio_data.info()
gunvio_data.shape
##using isnull to find out missing values
gunvio_data.isnull().values.any()
gunvio_data.isnull().sum()
#using isna to find out missing values
sum_missing_data= gunvio_data.isnull().sum()
sum_missing_data
count_missing_data=gunvio_data.isna().count()
count_missing_data
##Find percentage of missing data##
percentage_missing_data=(sum_missing_data/count_missing_data) * 100
percentage_missing_data
## Remove the blank data which is not useful for anaylsis####
gunvio_data.drop(["incident_characteristics",
              "latitude",
              'longitude',
              "incident_url",
              "sources",
              "source_url",
              "incident_url_fields_missing",
              "location_description",
              "participant_relationship",
              "notes",
    ], axis=1, inplace=True)
### https://stackoverflow.com/questions/21925114/is-there-an-implementation-of-missingmaps-in-pythons-ecosystem
from matplotlib import collections as collections
from matplotlib.patches import Rectangle
from itertools import cycle
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
        (fig, ax) = plt.subplots()

    # setup the axes

    dfn = pd.isnull(df)

    if sort in ('ascending', 'descending'):
        counts = dfn.sum()
        sort_dict = {'ascending': True, 'descending': False}
        counts = counts.sort_values(ascending=sort_dict[sort])
        dfn = dfn[counts.index]

    # Up to here

    ny = len(df)
    nx = len(df.columns)

    # each column is a stacked bar made up of ny patches.

    xgrid = np.tile(np.arange(nx), (ny, 1)).T
    ygrid = np.tile(np.arange(ny), (nx, 1))

    # xys is the lower left corner of each patch

    xys = (zip(x, y) for (x, y) in zip(xgrid, ygrid))

    if colors is None:
        colors = {True: '#EAF205', False: 'k'}

    widths = cycle([aspect])
    heights = cycle([1])

    for (xy, width, height, col) in zip(xys, widths, heights,
            dfn.columns):
        color_array = dfn[col].map(colors)

        rects = [Rectangle(xyc, width, height, **kwargs) for (xyc,
                 c) in zip(xy, color_array)]

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


colours = {True: "#fde725", False: "#440154"}
ax = missmap(gunvio_data, colors = colours)
plt.show(ax)
gunvio_data["date"] = pd.to_datetime(gunvio_data["date"])
gunvio_data["day"] = gunvio_data["date"].dt.day
gunvio_data["month"] = gunvio_data["date"].dt.month
gunvio_data["year"] = gunvio_data["date"].dt.year
gunvio_data["weekday"] = gunvio_data["date"].dt.weekday
gunvio_data["week"] = gunvio_data["date"].dt.week
gunvio_data["quarter"] = gunvio_data["date"].dt.quarter
gunvio_data.dtypes
gunvio_data["participant_gender"] = gunvio_data["participant_gender"].fillna("0::Unknown")
###dataset_gunviolence.participant_gender male_female.unique()####

def clean_participant_gender(row) :
    gender_row_values = []
    gender_row = str(row).split("||")
    for x in gender_row :
        gender_row_value = str(x).split("::")
        if len(gender_row_value) > 1 :
            gender_row_values.append(gender_row_value[1])
            
    return gender_row_values


participant_genders = gunvio_data.participant_gender.apply(clean_participant_gender)
gunvio_data["participant_gender_total"] = participant_genders.apply(lambda x: len(x))
gunvio_data["participant_gender_male"] = participant_genders.apply(lambda x: x.count("Male"))
gunvio_data["participant_gender_female"] = participant_genders.apply(lambda x: x.count("Female"))
gunvio_data["participant_gender_unknown"] = participant_genders.apply(lambda x: x.count("Unknown"))
del(participant_genders)
gunvio_data.columns.values
###dataset_gunviolence.quantiy stolen and not stolrn ####
gunvio_data["n_guns_involved"] = gunvio_data["n_guns_involved"].fillna(0)
gunvio_data["gun_stolen"] = gunvio_data["gun_stolen"].fillna("0::Unknown")
# Prints a lot but gives all the unique values of a column
#gunvio_data["gun_stolen"].unique()

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


gunstolenvalues = gunvio_data.gun_stolen.apply(clean_gun_stolen)
gunvio_data["gun_stolen_stolen"] = gunstolenvalues.apply(lambda x: x.count("Stolen"))
gunvio_data["gun_stolen_notstolen"] = gunstolenvalues.apply(lambda x: x.count("Not-stolen"))
del(gunstolenvalues)
gunvio_data.columns.values
gunvio_data["n_guns_involved"] = gunvio_data["n_guns_involved"].fillna(0)
gunvio_data["gun_stolen"] = gunvio_data["gun_stolen"].fillna("0::Unknown")
# ###Prints a lot but gives all the unique values of a column
#gunvio_data["gun_stolen"].unique()

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


gunstolenvalues = gunvio_data.gun_stolen.apply(clean_gun_stolen)
gunvio_data["gun_stolen_stolen"] = gunstolenvalues.apply(lambda x: x.count("Stolen"))
gunvio_data["gun_stolen_notstolen"] = gunstolenvalues.apply(lambda x: x.count("Not-stolen"))
del(gunstolenvalues)
# Check values for new columns added
gunvio_data.head()
gunvio_data.shape


###DATA EXPLORATION WITH THE HELP OF GRAPHS

##The following graphs have been used to describe the gun violence data :

   # i) Joint Distribution plots
   # ii) Histograms
    #iii) Pie chart
  #  iv) Kernel Density plots
    #v) Point plots
    #vi) Violin plots
   # vii) Box plots
#viii) Count plots
    #ix) Facet Grid plots


####Jointplot between Number of Person Killed Vs Injured in all incidences##
sns.jointplot("n_injured",
              "n_killed",
              gunvio_data,
              kind='scatter',      # kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }, optional
              s=200, color='m', edgecolor="skyblue", linewidth=2)
# Jointplot to identify Maximum Number of Person Injured in which incidence
sns.jointplot("incident_id",
              "n_injured",
              gunvio_data,
              kind='scatter'      # kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }, optional
              )
###Jointplot to identify Maximum Number of Person Killed in which incidence###
sns.jointplot("incident_id",
              "n_killed",
              gunvio_data,
              kind='scatter',      # kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }, optional
              color="Red",
              marginal_kws={'color': 'red'})
# Jointplot to identify the number guns involved and the number of guns stolen
sns.jointplot(x=gunvio_data["n_guns_involved"], y=gunvio_data["gun_stolen_stolen"], kind="scatter", color="#D81B60")
# Jointplot to identify the number of people killed based on gender wise participant total
sns.jointplot(x=gunvio_data.participant_gender_total, y=gunvio_data.n_killed, data=gunvio_data, space=0, dropna=True, color="#D81B60")
# Jointplot to identify the number of people injured based on gender wise participant total
sns.jointplot(x=gunvio_data.participant_gender_total, y=gunvio_data.n_injured, data=gunvio_data, space=0, dropna=True, color="#1E88E5")
# Histogram for Top 10 Cities with maximum incidents of Gun Violence
citywise_total = gunvio_data[["incident_id"]].groupby(gunvio_data["city_or_county"]).count()
top_cities = citywise_total.sort_values(by='incident_id', ascending=False).head(10)
print(top_cities)
top_cities.plot.barh()
del(top_cities)
# Histogram for Top 10 States with maximum incidents of Gun Violence
statewise_total = gunvio_data[["incident_id"]].groupby(gunvio_data["state"]).count()
top_states = statewise_total.sort_values(by='incident_id', ascending=False).head(10)
print(top_states)
top_states.plot.barh()
del(top_states)
# Histogram for Weekday wise Incidents
weekwise_total = gunvio_data[["incident_id"]].groupby(gunvio_data["weekday"]).count()
weekwise_total.plot.barh()
del(weekwise_total)
# Here, for weekdays 0 is for Monday and 6 is for Sunday.
# Histogram showing the crime rate state wise
state_vs_crimecount=sns.countplot(x=gunvio_data["state"],data=gunvio_data,order=gunvio_data["state"].value_counts().index)
state_vs_crimecount.set_xticklabels(state_vs_crimecount.get_xticklabels(),rotation=90)
state_vs_crimecount.set_title("State Vs Crime Rate")
# Histogram showing the the top 10 cities with high crime rate
city_vs_crimerate=gunvio_data['city_or_county'].value_counts().head(10)
city_vs_crimerate=sns.barplot(x=city_vs_crimerate.index,y=city_vs_crimerate)
city_vs_crimerate.set_xticklabels(city_vs_crimerate.get_xticklabels(),rotation=45)
city_vs_crimerate.set_title("Top 10 Cities having high crime rate")
# state wise crime count for top 10 states
state_vs_crimecount=gunvio_data['state'].value_counts().head(10)
state_vs_crimecount
# Pie chart showing Top 10 States having high crime rate
plt.pie(state_vs_crimecount,labels=state_vs_crimecount.index,shadow=True)
plt.title("Top 10 States having high crime rate")
plt.axis("equal")
# the same graph in another way
plt.pie(state_vs_crimecount, labels=state_vs_crimecount.index,autopct="%1.1f%%", shadow=True, startangle=195)
plt.title("Top 10 States having high crime rate")
plt.axis("equal")
# state wise crime rate for all the states
statewise_crime_rate = gunvio_data["state"].value_counts()
statewise_crime_rate
# Pie chart showing state wise crime rate for all the states
plt.pie(statewise_crime_rate, labels=statewise_crime_rate.index,autopct="%1.1f%%", shadow=True, startangle=195)
plt.title("State-wise Gun Violence Percentage")
plt.axis("equal")
# Top 50 cities having highest crime rate
topcitywise_crime_rate = gunvio_data["city_or_county"].value_counts().head(50)
topcitywise_crime_rate
# Pie chart showing Top 50 cities having highest crime rate
plt.pie(topcitywise_crime_rate, labels=topcitywise_crime_rate.index,autopct="%1.1f%%", shadow=True, startangle=195)
plt.title("City-wise Gun Violence Percentage")
plt.axis("equal")
### Kernal Density plot for genderwise participant###
genderwise_total = gunvio_data[["participant_gender_total", "participant_gender_male", "participant_gender_female", "participant_gender_unknown"]].groupby(gunvio_data["year"]).sum()
dp_gen_plot=sns.kdeplot(genderwise_total["participant_gender_male"], shade=True, color="r")
dp_gen_plot=sns.kdeplot(genderwise_total["participant_gender_female"], shade=True, color="b")
dp_gen_plot=sns.kdeplot(genderwise_total['participant_gender_unknown'], shade=True, color="g")
del(genderwise_total)
###Density plot for person injured vs killed on all weekdays##
inj_kill_weektotal = gunvio_data[["n_injured","n_killed"]].groupby(gunvio_data["weekday"]).sum()
dp_inj_kill_plot=sns.kdeplot(inj_kill_weektotal['n_injured'], shade=True, color="r")
dp_inj_kill_plot=sns.kdeplot(inj_kill_weektotal['n_killed'], shade=True, color="b")
del(inj_kill_weektotal)
###Violin Plot shows Year wise no of people injured###
year_vs_injured_plot = sns.violinplot("year", "n_injured", data=gunvio_data,split=True, inner="quartile")
year_vs_injured_plot.set_title("Persons injured in the incidents per Year")
###Violin Plot for Year wise no of people injured
year_vs_injured_plot = sns.violinplot("year", "n_injured", data=gunvio_data,split=True, inner="quartile")
year_vs_injured_plot.set_title("Persons injured in the incidents per Year")
gunvio_data.columns.values
# Created a new column for the total number of persons impacted (injured+killed) as per the data available
gunvio_data["total_impacted"] = gunvio_data["n_killed"] + gunvio_data["n_injured"]
# Violin Plot for total num of persons Impacted(Killed/Injured) during gun violence
Impacted_persons_total = gunvio_data[["total_impacted", "n_injured", "n_killed"]].groupby(gunvio_data["year"]).sum()
print(Impacted_persons_total)
year_impacted_plot = sns.violinplot(data=Impacted_persons_total,
               split=True,         # If hue variable has two levels, draw half of a violin for each level.
               inner="quartile"    #  Options: “box”, “quartile”, “point”, “stick”, None 
               )
year_impacted_plot.set_title("Total poeple Impacted(Killed/Injured) due to gun violence")
del(Impacted_persons_total)
# Violin Plot for Genderwise total number of persons involved/impacted during gun violence
genderwise_total = gunvio_data[["participant_gender_total", "participant_gender_male", "participant_gender_female", "participant_gender_unknown"]].groupby(gunvio_data["year"]).sum()
print(genderwise_total)
year_genderwise_plot = sns.violinplot(data=genderwise_total,
               split=True,         # If hue variable has two levels, draw half of a violin for each level.
               inner="quartile"    #  Options: “box”, “quartile”, “point”, “stick”, None 
               )
year_genderwise_plot.set_title("Genderwise total number of persons involved/impacted during gun violence")
del(genderwise_total)
###State Vs No of People Killed###
statewise_vs_killed=gunvio_data.groupby(gunvio_data["state"]).apply(lambda x: pd.Series(dict(No_Killed=x.n_killed.sum())))
statewise_vs_killed
# Box plot for total number of persons killed State wise
sns.boxplot('state','n_killed',data=gunvio_data)
#State Vs No of people Injured####
statewise_vs_injured=gunvio_data.groupby(gunvio_data["state"]).apply(lambda x: pd.Series(dict(No_Injured=x.n_injured.sum())))
statewise_vs_injured
# Box plot for total number of persons injured State wise
sns.boxplot('state','n_injured',data=gunvio_data)
# Box Plot for Monthwise total number of Persons Injured
month_injured_plot = sns.boxplot("month", "n_injured", data= gunvio_data)
month_injured_plot.set_title("Person injured in incidents per month")
# Count Plot for Statewise incidences of Gun Violence
statewise_inc_plot = sns.countplot(x=gunvio_data["state"], data = gunvio_data,order=gunvio_data["state"].value_counts().index)
statewise_inc_plot.set_title("Statewise incidence of Gun Violence")
statewise_inc_plot.set_xticklabels(statewise_inc_plot.get_xticklabels(), rotation=90)
# Count Plot for State Senate District wise
state_incident_plot = sns.countplot("state_senate_district", data = gunvio_data,order=gunvio_data["state_house_district"].value_counts().index)
state_incident_plot.set_title("State Senate District wise incidence of Gun Violence")
state_incident_plot.set_xticklabels(state_incident_plot.get_xticklabels(),rotation=90)
# Facet Grid Graph for Male/ Female Partipant per Year
g = sns.FacetGrid(gunvio_data, hue="year", palette="Set1", size=5, hue_kws={"marker": ["^", "v","*",">","<","+"]})
g.map(plt.scatter, "participant_gender_male", "participant_gender_female", s=100, linewidth=.5, edgecolor="white")
g.add_legend();
# Facet Grid Graphh for Person killed and Injured on Particular days of the week
g = sns.FacetGrid(gunvio_data, hue="weekday", palette="Set1", size=5, hue_kws={"marker": ["^", "v","h","o","+",">","d"]})
g.map(plt.scatter, "n_injured", "n_killed", s=100, linewidth=.5, edgecolor="white")
g.add_legend();
