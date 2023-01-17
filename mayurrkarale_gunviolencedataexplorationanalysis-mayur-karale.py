# Import numpy and pandas
import numpy as np
import pandas as pd
import os, sys, time
# Import matplotlib for plotting (matplotlib library for the Python programming language and its numerical mathematics extension NumPy)
import matplotlib as mpl
import matplotlib.pyplot as plt   
# Import seaborn data visualization library (It is a based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
import seaborn as sns # Easier plotting
# To show graphs in same window
%matplotlib inline
# Set matplotlib and seaborn map styles
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
# Set working directory
print(os.listdir("../input"))
#inputFolder = "E:\\BigDataAnalytics\\MayurAssignments\\Gun Violence Data Exploration Analysis using python\\"
#os.chdir(inputFolder)

#OR
#os.getcwd()
#os.chdir("E:\\BigDataAnalytics\\MayurAssignments\\Gun Violence Data Exploration Analysis using python")
#os.listdir()  # Directory contents
# Read data from fifa_ranking.csv file
#gunviolencedata = pd.read_csv('gun-violence-data_01-2013_03-2018.csv')

#OR
#gunviolencedata = pd.read_csv('E:\\BigDataAnalytics\\MayurAssignments\\Gun Violence Data Exploration Analysis using python\\gun-violence-data_01-2013_03-2018.csv')

gunviolencedata = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")
gunviolencedata.head()
# Print full summary
#Meta info
gunviolencedata.info()
# How many columns and rows
gunviolencedata.shape
# How many rows?
gunviolencedata.shape[0]
# How many columns?
gunviolencedata.shape[1]
# Get the row names
gunviolencedata.index.values
# Get the column names
gunviolencedata.columns.values
# Check datatypes
gunviolencedata.dtypes
 # Explore gunviolancedata 
gunviolencedata.describe()

# Check using isnull to find out missing values in gunviolencedata
gunviolencedata.isnull().values.any()
# Check gunviolencedata columns where we dont have missing values
gunviolencedata.isnull().sum()
# Check gunviolencedata columns count
gunviolencedatacount=gunviolencedata.isna().count()
gunviolencedatacount
# Check gunviolencedata columns count where we have missing values
gunviolencedatamissingcount=gunviolencedata.isna().sum()
gunviolencedatamissingcount
# Calculate the % of missing data in gunviolencedata
gunviolencemissingdatapercentage =(gunviolencedatamissingcount/gunviolencedatacount) * 100
gunviolencemissingdatapercentage
gunviolencedata.shape
gunviolencemissingdata = pd.concat([gunviolencedatamissingcount, gunviolencemissingdatapercentage], axis=1)
gunviolencemissingdata
#Delete the data created
del(gunviolencedatamissingcount,gunviolencedatacount,gunviolencemissingdatapercentage)
gunviolencemissingdata
# Check using isnull to find out missing values in gunviolencedata
gunviolencedata.isnull().values.any()
# https://stackoverflow.com/questions/21925114/is-there-an-implementation-of-missingmaps-in-pythons-ecosystem
from matplotlib import collections as collections
from matplotlib.patches import Rectangle

# Import intertools library
import itertools
zip = getattr(itertools, 'izip', zip)
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
ax = missmap(gunviolencedata, colors = colours)
plt.show(ax)
# Check datatypes
gunviolencedata.dtypes
# Convert object datatype to datetime
gunviolencedata["date"] = pd.to_datetime(gunviolencedata["date"])
gunviolencedata["day"] = gunviolencedata["date"].dt.day
gunviolencedata["weekday"] = gunviolencedata["date"].dt.weekday
gunviolencedata["week"] = gunviolencedata["date"].dt.week
gunviolencedata["quarter"] = gunviolencedata["date"].dt.quarter
gunviolencedata["month"] = gunviolencedata["date"].dt.month
gunviolencedata["year"] = gunviolencedata["date"].dt.year
#Check the datatype of columns again
gunviolencedata.dtypes
# Check values for gun_type
gunviolencedata["gun_type"].unique()
# Created a new column for the total number of persons impacted (injured+killed) as per the gunviolencedata
gunviolencedata["total_impacted"] = gunviolencedata["n_killed"] + gunviolencedata["n_injured"]
gunviolencedata.columns.values
#check count of columns and rows after adding new columns
gunviolencedata.shape
# Creating multiple columns from Participant's Gender column
gunviolencedata["participant_gender"] = gunviolencedata["participant_gender"].fillna("0::Unknown")

def clean_participant_gender(row) :
    gender_row_values = []
    gender_row = str(row).split("||")
    for x in gender_row :
        gender_row_value = str(x).split("::")
        if len(gender_row_value) > 1 :
            gender_row_values.append(gender_row_value[1])
            
    return gender_row_values

participant_genders = gunviolencedata.participant_gender.apply(clean_participant_gender)
gunviolencedata["participant_gender_total"] = participant_genders.apply(lambda x: len(x))
gunviolencedata["participant_gender_male"] = participant_genders.apply(lambda x: x.count("Male"))
gunviolencedata["participant_gender_female"] = participant_genders.apply(lambda x: x.count("Female"))
gunviolencedata["participant_gender_unknown"] = participant_genders.apply(lambda x: x.count("Unknown"))
del(participant_genders)


# Check newly added column names
gunviolencedata.columns.values
#check count of columns and rows after adding new columns
gunviolencedata.shape
# Check for null value of column for guns involved and guns stolen and filling the missing values
gunviolencedata["n_guns_involved"] = gunviolencedata["n_guns_involved"].fillna(value =0)
gunviolencedata["gun_stolen"] = gunviolencedata["gun_stolen"].fillna(value = "0::Unknown")
# Prints a lot but gives all the unique values of a column, gunviolencedata["gun_stolen"].unique()

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


gunstolenvalues = gunviolencedata.gun_stolen.apply(clean_gun_stolen)
gunviolencedata["gun_stolen_stolen"] = gunstolenvalues.apply(lambda x: x.count("Stolen"))
gunviolencedata["gun_stolen_notstolen"] = gunstolenvalues.apply(lambda x: x.count("Not-stolen"))
del(gunstolenvalues)
# Check newly added column names
gunviolencedata.columns.values
#check count of columns and rows after adding new columns
gunviolencedata.shape
#Similar way create different age groups
#gunviolencedata.participant_age_group.unique()
gunviolencedata["participant_age_group"] = gunviolencedata["participant_age_group"].fillna("0::Unknown")

def clean_participant_age_group(row) :
    unknownCount = 0
    childCount = 0
    teenCount = 0
    adultCount = 0
    agegroup_row_values = []
    
    agegroup_row = str(row).split("||")
    for x in agegroup_row :
        agegroup_row_value = str(x).split("::")
        if len(agegroup_row_value) > 1 :
            agegroup_row_values.append(agegroup_row_value[1])
            if "Child 0-10" in agegroup_row_value :
                childCount += 1
            elif "Teen 11-17" in agegroup_row_value :
                teenCount += 1
            elif "Adult 18+" in agegroup_row_value :
                adultCount += 1
            else :
                unknownCount += 1
                
    return agegroup_row_values

age_group = gunviolencedata.participant_age_group.apply(clean_participant_age_group)
gunviolencedata["agegroup_child"] = age_group.apply(lambda x: x.count("Child 0-10"))
gunviolencedata["agegroup_teen"] = age_group.apply(lambda x: x.count("Teen 11-17"))
gunviolencedata["agegroup_adult"] = age_group.apply(lambda x: x.count("Adult 18+"))
del(age_group)
# Check newly added column names
gunviolencedata.columns.values
#check count of columns and rows after adding new columns
gunviolencedata.shape
#Joint Distribution plots
# Jointplot between Number of Person Killed vs Injured
sns.jointplot("n_injured",
              "n_killed",
              gunviolencedata,
              kind='scatter',      # kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }, optional
              s=200, color='blue', edgecolor="red", linewidth=2)
# Jointplot between number of guns invovled vs gun_stolen_stolen
sns.jointplot(x= gunviolencedata["n_guns_involved"], y= gunviolencedata["gun_stolen_stolen"], kind="scatter", color='purple')
# Jointplot between number of guns invovled vs gun_stolen_notstolen
sns.jointplot(x=gunviolencedata["n_guns_involved"], y= gunviolencedata["gun_stolen_notstolen"], kind="scatter", color="#1E88E5")
# Jointplot to identify Maximum Number of Person Injured
sns.jointplot("incident_id",
              "n_injured",
              gunviolencedata,
              color ='purple',
              kind='scatter'      # kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }, optional
              )
# Jointplot to identify Maximum Number of Person Killed 
sns.jointplot("incident_id",
              "n_killed",
              gunviolencedata,
              kind='scatter',      # kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }, optional
              color="skyblue",
              marginal_kws={'color': 'red'})
# Jointplot to identify the number of people killed based on gender wise participant total
sns.jointplot(x=gunviolencedata.participant_gender_total, y=gunviolencedata.n_killed, data=gunviolencedata, space=0, dropna=True, color='skyblue')
# Jointplot to identify the number of people injured based on gender wise participant total
sns.jointplot(x=gunviolencedata.participant_gender_total, y=gunviolencedata.n_injured, data=gunviolencedata, space=0, dropna=True, color='red')
# Histograms
# Histogram to show the citi wise incidents
citywisetotal = gunviolencedata[["incident_id"]].groupby(gunviolencedata["city_or_county"]).count()
top_cities = citywisetotal.sort_values(by='incident_id', ascending=False).head(10)
print(top_cities)
top_cities.plot.barh()
del(top_cities)
# Histogram to show the state wise incidents
statewisetotal = gunviolencedata[["incident_id"]].groupby(gunviolencedata["state"]).count()
top_states = statewisetotal.sort_values(by='incident_id',ascending=False).head(10)
print(top_states)
top_states.plot.barh()
del(top_states)
# Histogram to show the the top 10 cities with high crime rate
cityvscrimerate=gunviolencedata['city_or_county'].value_counts().head(10)
cityvscrimerate=sns.barplot(x=cityvscrimerate.index,y=cityvscrimerate)
cityvscrimerate.set_xticklabels(cityvscrimerate.get_xticklabels(),rotation=45)
cityvscrimerate.set_title("Top 10 Cities having high crime rate")
# Histogram to show the state wise crime rate
statewisecrimerate = sns.countplot(x=gunviolencedata["state"], data=gunviolencedata, palette=btui, order=gunviolencedata["state"].value_counts().index)
statewisecrimerate.set_xticklabels(statewisecrimerate.get_xticklabels(), rotation=90)
statewisecrimerate.set_title("State(s) with highest number of Gun Violence")
# Histogram to show the the top cities with high crime rate
cityvscrimerate=gunviolencedata['city_or_county'].value_counts().head(40)
cityvscrimerate=sns.barplot(x=cityvscrimerate.index,y=cityvscrimerate)
cityvscrimerate.set_xticklabels(cityvscrimerate.get_xticklabels(),rotation=45)
cityvscrimerate.set_title("Cities or Counties with highest number of Gun Violence")
fig, axs = plt.subplots(ncols=2)
n_crimes_by_month_k = gunviolencedata.groupby(gunviolencedata["month"]).apply(lambda x: pd.Series(dict(total_killed_per_month = x.n_killed.sum())))
n_crimes_by_month_plot_k = sns.barplot(x=n_crimes_by_month_k.index, y=n_crimes_by_month_k.total_killed_per_month, palette=btui, ax=axs[0])
n_crimes_by_month_plot_k.set_title("Killed per Month")
del(n_crimes_by_month_k)

n_crimes_by_month_i = gunviolencedata.groupby(gunviolencedata["month"]).apply(lambda x: pd.Series(dict(total_injured_per_month = x.n_injured.sum())))
n_crimes_by_month_plot_i = sns.barplot(x=n_crimes_by_month_i.index, y=n_crimes_by_month_i.total_injured_per_month, palette=btui, ax=axs[1])
n_crimes_by_month_plot_i.set_title("Injured per Month")
del(n_crimes_by_month_i)

fig, axs = plt.subplots(ncols=2)
n_crimes_by_year_k = gunviolencedata.groupby(gunviolencedata["year"]).apply(lambda x: pd.Series(dict(total_killed_per_year = x.n_killed.sum())))
n_crimes_by_year_plot_k = sns.barplot(x=n_crimes_by_year_k.index, y=n_crimes_by_year_k.total_killed_per_year, palette=btui, ax=axs[0])
n_crimes_by_year_plot_k.set_title("Killed each year")
del(n_crimes_by_year_k)

n_crimes_by_year_i = gunviolencedata.groupby(gunviolencedata["year"]).apply(lambda x: pd.Series(dict(total_injured_per_year = x.n_injured.sum())))
n_crimes_by_year_plot_i = sns.barplot(x=n_crimes_by_year_i.index, y=n_crimes_by_year_i.total_injured_per_year, palette=btui, ax=axs[1])
n_crimes_by_year_plot_i.set_title("Injured each year")
del(n_crimes_by_year_i)
#Pie charts
# State wise gun violence %
statewisecrimerate = gunviolencedata["state"].value_counts()
statewisecrimerate
plt.pie(statewisecrimerate, labels=statewisecrimerate.index, colors=btui, autopct="%1.1f%%", shadow=True, startangle=195)
plt.title("State wise Gun Violence Percentage")
plt.axis("equal")
# City or County wise gun violence %
cityorcountywisecrimerate = gunviolencedata["city_or_county"].value_counts().head(40)
cityorcountywisecrimerate
plt.pie(cityorcountywisecrimerate, labels=cityorcountywisecrimerate.index, colors=btui, autopct="%1.1f%%", shadow=True, startangle=195)
plt.title("City or County wise Gun Violence Percentage")
plt.axis("equal")
# Top 10 State wise gun violence %
statewisecrimerate = gunviolencedata["state"].value_counts().head(10)
statewisecrimerate
plt.pie(statewisecrimerate, labels=statewisecrimerate.index, colors=btui, autopct="%1.1f%%", shadow=True, startangle=195)
plt.title("Top 10 State wise Gun Violence Percentage")
plt.axis("equal")
# Top 10 City or County wise gun violence %
cityorcountywisecrimerate = gunviolencedata["city_or_county"].value_counts().head(10)
cityorcountywisecrimerate
plt.pie(cityorcountywisecrimerate, labels=cityorcountywisecrimerate.index, colors=btui, autopct="%1.1f%%", shadow=True, startangle=195)
plt.title("Top 10 City or County wise Gun Violence Percentage")
plt.axis("equal")
# state wise crime count for top 10 states
statevscrimecount=gunviolencedata['state'].value_counts().head(10)
statevscrimecount

# Pie chart to show Top 10 States having high crime rate
plt.pie(statevscrimecount,labels=statevscrimecount.index,shadow=True, colors=btui,startangle=195,autopct="%1.1f%%")
plt.title("Top 10 States having high crime rate")
plt.axis("equal")
# City or County wise crime count for top 10 states
cityorcountycrimecount=gunviolencedata['city_or_county'].value_counts().head(10)
cityorcountycrimecount
plt.pie(cityorcountycrimecount,labels=cityorcountycrimecount.index,shadow=True,colors=btui,startangle=195,autopct="%1.1f%%")
plt.title("Top 10 City or County having high crime rate")
plt.axis("equal")
# Kernel Density plots
# https://seaborn.pydata.org/examples/multiple_joint_kde.html
genderwisetotal = gunviolencedata[["participant_gender_total", "participant_gender_male", "participant_gender_female", "participant_gender_unknown"]].groupby(gunviolencedata["year"]).sum()
genderplot=sns.kdeplot(genderwisetotal["participant_gender_male"], shade=True, color="red")
genderplot=sns.kdeplot(genderwisetotal["participant_gender_female"], shade=True, color="blue")
genderplot=sns.kdeplot(genderwisetotal['participant_gender_unknown'], shade=True, color="green")
del(genderwisetotal)
# https://seaborn.pydata.org/examples/multiple_joint_kde.html
injuredkilledweektotal = gunviolencedata[["n_injured","n_killed"]].groupby(gunviolencedata["weekday"]).sum()
injuredkilledplot=sns.kdeplot(injuredkilledweektotal['n_injured'], shade=True, color="purple")
injuredkilledplot=sns.kdeplot(injuredkilledweektotal['n_killed'], shade=True, color="green")
del(injuredkilledweektotal)

# https://seaborn.pydata.org/examples/multiple_joint_kde.html
injuredkilledmonthtotal = gunviolencedata[["n_injured","n_killed"]].groupby(gunviolencedata["month"]).sum()
injuredkilledplot1=sns.kdeplot(injuredkilledmonthtotal['n_injured'], shade=True, color="purple")
injuredkilledplot1=sns.kdeplot(injuredkilledmonthtotal['n_killed'], shade=True, color="green")
del(injuredkilledmonthtotal)
# Point Plots
yearlywiseinjured=gunviolencedata.groupby(gunviolencedata["year"]).apply(lambda x: pd.Series(dict(No_Injured=x.n_injured.sum())))
yearlywiseinjured_plot=sns.pointplot(x=yearlywiseinjured.index, y=yearlywiseinjured.No_Injured, data=yearlywiseinjured,label="yearlywiseinjured")
yearlywiseinjured
yearlywisekilled=gunviolencedata.groupby(gunviolencedata["year"]).apply(lambda x: pd.Series(dict(No_Killed=x.n_killed.sum())))
yearlywisekilled_plot=sns.pointplot(x=yearlywisekilled.index, y=yearlywisekilled.No_Killed, data=yearlywisekilled,label="yearlywisekilled")
yearlywisekilled
monthlywiseinjured=gunviolencedata.groupby(gunviolencedata["month"]).apply(lambda x: pd.Series(dict(No_Injured=x.n_injured.sum())))
monthlywiseinjured_plot=sns.pointplot(x=monthlywiseinjured.index, y=monthlywiseinjured.No_Injured, data=monthlywiseinjured,label="monthlywiseinjured")
monthlywiseinjured
monthlywisekilled=gunviolencedata.groupby(gunviolencedata["month"]).apply(lambda x: pd.Series(dict(No_Killed=x.n_killed.sum())))
monthlywisekilled_plot=sns.pointplot(x=monthlywisekilled.index, y=monthlywisekilled.No_Killed, data=monthlywisekilled,label="monthlywisekilled")
monthlywisekilled
agegroupsum = gunviolencedata[["state","agegroup_child", "agegroup_teen", "agegroup_adult"]].groupby(gunviolencedata["state"]).sum()
g=sns.pointplot(x=agegroupsum.index, y=agegroupsum.agegroup_child, data=agegroupsum, color="#c51b7d", scale=0.5, dodge=True, capsize=.2, label="agegroup_adult")
g=sns.pointplot(x=agegroupsum.index, y=agegroupsum.agegroup_teen, data=agegroupsum, color="#1b7837", scale=0.5, dodge=True, capsize=.2, linestyles="--", markers="x", label="agegroup_adult")
g=sns.pointplot(x=agegroupsum.index, y=agegroupsum.agegroup_adult, data=agegroupsum, color="#2166ac", scale=0.5, dodge=True, capsize=.2, linestyles="-", markers="X", label="agegroup_adult")
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_title("Gun Violence per various Age Group")
plt.show()
# Violin plots
# Violin Plot for Year wise no of people killed
yearwisekilledviolinplot = sns.violinplot("year", "n_killed",
               data=gunviolencedata,
               split=True,         # If hue variable has two levels, draw half of a violin for each level.
               inner="quartile"    #  Options: “box”, “quartile”, “point”, “stick”, None 
               )
yearwisekilledviolinplot.set_title("Persons killed in the incidents per Year")
# Violin Plot for Year wise no of people injured
yearwiseinjuredviolinplot = sns.violinplot("year", "n_injured", data=gunviolencedata,split=True, inner="quartile")
yearwiseinjuredviolinplot.set_title("Persons injured in the incidents per Year")
# Violin Plot for Month wise no of people killed
monthwisekilledviolinplot = sns.violinplot("month", "n_killed",
               data=gunviolencedata,
               split=True,         # If hue variable has two levels, draw half of a violin for each level.
               inner="quartile"    #  Options: “box”, “quartile”, “point”, “stick”, None 
               )
monthwisekilledviolinplot.set_title("Persons killed in the incidents per month")
# Violin Plot for Month wise no of people injured
monthwiseinjuredviolinplot = sns.violinplot("month", "n_injured",
               data=gunviolencedata,
               split=True,         # If hue variable has two levels, draw half of a violin for each level.
               inner="quartile"    #  Options: “box”, “quartile”, “point”, “stick”, None 
               )
monthwiseinjuredviolinplot.set_title("Persons injured in the incidents per month")
# Violin Plot for total num of persons Impacted(Killed/Injured) during gun violence
totalimpactedpersons = gunviolencedata[["total_impacted", "n_injured", "n_killed"]].groupby(gunviolencedata["year"]).sum()
print(totalimpactedpersons)
yearlyimpactedviolinplot = sns.violinplot(data=totalimpactedpersons,
               split=True,         # If hue variable has two levels, draw half of a violin for each level.
               inner="quartile"    #  Options: “box”, “quartile”, “point”, “stick”, None 
               )
yearlyimpactedviolinplot.set_title("Total number of persons Impacted(Killed/Injured) during gun violence")
del(totalimpactedpersons)
# Violin Plot for Genderwise total number of persons involved/impacted during gun violence
genderwisetotal = gunviolencedata[["participant_gender_total", "participant_gender_male", "participant_gender_female", "participant_gender_unknown"]].groupby(gunviolencedata["year"]).sum()
print(genderwisetotal)
yeargenderwiseplot = sns.violinplot(data=genderwisetotal,
               split=True,         # If hue variable has two levels, draw half of a violin for each level.
               inner="quartile"    #  Options: “box”, “quartile”, “point”, “stick”, None 
               )
yeargenderwiseplot.set_title("Genderwise total number of persons involved/impacted during gun violence")
del(genderwisetotal)
# Box Plots
#Statewise number of people Injured
statewiseinjured=gunviolencedata.groupby(gunviolencedata["state"]).apply(lambda x: pd.Series(dict(No_Injured=x.n_injured.sum())))
statewiseinjured

# Box plot for total number of persons Injured State wise
sns.boxplot('state','n_injured',data=gunviolencedata)
# State wise number of People Killed
statewisekilled=gunviolencedata.groupby(gunviolencedata["state"]).apply(lambda x: pd.Series(dict(No_Killed=x.n_killed.sum())))
statewisekilled
# Box plot for total number of persons killed State wise
sns.boxplot('state','n_killed',data=gunviolencedata)
# Box plot for total number of persons injured per month
monthwiseinjuredboxplot = sns.boxplot("month", "n_injured", data= gunviolencedata)
monthwiseinjuredboxplot.set_title("Person injured in incidents per month")
# Box plot for total number of persons killed per month
monthwisekilledboxplot = sns.boxplot("month", "n_killed", data= gunviolencedata)
monthwisekilledboxplot.set_title("Person killed in incidents per month")
# Box plot for total number of persons injured per year
yearwiseinjuredboxplot = sns.boxplot("year", "n_injured", data= gunviolencedata)
yearwiseinjuredboxplot.set_title("Person injured in incidents per year")
# Box plot for total number of persons killed per year
yearwisekilledboxplot = sns.boxplot("year", "n_killed", data= gunviolencedata)
yearwisekilledboxplot.set_title("Person killed in incidents per year")
participantgenderssum = gunviolencedata[["state", "participant_gender_total", "participant_gender_male", "participant_gender_female", "participant_gender_unknown"]].groupby(gunviolencedata["state"]).sum()
sns.boxplot(data=participantgenderssum, palette=btui)
g=sns.barplot(x=participantgenderssum.index,y=participantgenderssum.participant_gender_total,data=participantgenderssum,color='skyblue')
g=sns.barplot(x=participantgenderssum.index,y=participantgenderssum.participant_gender_male,data=participantgenderssum,color='yellow')
g=sns.barplot(x=participantgenderssum.index,y=participantgenderssum.participant_gender_female,data=participantgenderssum,color='green')
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.show()
# Count Plots
statewiseincidentcountplot = sns.countplot(x=gunviolencedata["state"], data = gunviolencedata,palette=btui,order=gunviolencedata["state"].value_counts().index)
statewiseincidentcountplot.set_title("Statewise incident of Gun Violence")
statewiseincidentcountplot.set_xticklabels(statewiseincidentcountplot.get_xticklabels(), rotation=90)
statewisecrimeratecountplot = sns.countplot(x=gunviolencedata["state"], data=gunviolencedata, palette=btui, order=gunviolencedata["state"].value_counts().index)
statewisecrimeratecountplot.set_xticklabels(statewisecrimeratecountplot.get_xticklabels(), rotation=90)
statewisecrimeratecountplot.set_title("State(s) with highest number of Gun Violence")
# Facet grid graph
# Facet Grid Graphh for Person killed and Injured on Particular days of the week
g = sns.FacetGrid(gunviolencedata, hue="weekday", palette="Set1", size=5, hue_kws={"marker": ["^", "v","h","o","+",">","d"]})
g.map(plt.scatter, "n_injured", "n_killed", s=100, linewidth=.5, edgecolor="white")
g.add_legend();
# Facet Grid Graph for Male/ Female Partipant per Year
g = sns.FacetGrid(gunviolencedata, hue="year", palette="Set1", size=5, hue_kws={"marker": ["^", "v","*",">","<","."]})
g.map(plt.scatter, "participant_gender_male", "participant_gender_female", s=100, linewidth=.5, edgecolor="white")
g.add_legend();
# Facet Grid Graphh for Person killed and Injured per Year
g = sns.FacetGrid(gunviolencedata, hue="year", palette="Set1", size=5, hue_kws={"marker": ["^", "v","*",">","<","o"]})
g.map(plt.scatter, "n_injured", "n_killed", s=100, linewidth=.5, edgecolor="white")
g.add_legend();
# Heatmap
sns.heatmap(gunviolencedata.corr(), cmap=btui, annot=True, fmt=".2f")