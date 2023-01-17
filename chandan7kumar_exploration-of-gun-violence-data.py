################################################################################################################################
#Last amended:16th August, 2018
#My folder: C:\ML\FORE\EXERCISE_IN_PYTHON\GunViolenceDataUsingPython
#Data file: gun-violence-data_01-2013_03-2018.csv

#Ref:
#    1. https://www.kaggle.com/jameslko/gun-violence-data/home
#    
#Description : Gun Violence Data which holds comprehensive record of over 260k US gun violence incidents
#              from 2013-2018. The below fields are available in the dataset for analysis.
#              Column:
#                  incident_id, Date of crime, State of crime, City/ County of crime,
#                  Address of the location of the crime, n_killed, n_injured, URL regarding the incident,
#                  source_url, incident_url_fields_missing, congressional_district,
#                  Status of guns involved in the crime (i.e. Unknown, Stolen, etc...), 
#                 gun_type (Typification of guns used in the crime), incident_characteristics, 
#                  (latitude) Location of the incident, location_description, 
#                  (longitude) Location of the incident, 
#                  n_guns_involved (Number of guns involved in incident), 
#                  notes (Additional information of the crime), participant_age, 
#                  participant_age_group, participant_gender, participant_name,
#                  participant_relationship, Extent of harm done to the participant,
#                  Type of participant, sources, state_house_district,state_senate_district
#              
#Objectives: Draw the below graphs.
#            i)  Joint Distribution plots
#            ii)  Histograms
#            iii) Kernel Density plots
#            iv) Violin plots
#            v) Box plots
#            vi) FacetGrid (see this link) 
#             
################################################################################################################################
# Importing libraries 
import pandas as pd        # R-like data manipulation
import numpy as np         # n-dimensional arrays
import matplotlib as mpl
import matplotlib.pyplot as plt      # For base plotting
import seaborn as sns                # Easier plotting
%matplotlib inline
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
#Read data file
data = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")
#Explore data
data.columns.values 
data.dtypes
data.describe() 
data.info()
data.shape
data.head()
# using isnull to find out missing values
data.isnull().values
data.isnull().values.any()
data.isnull().sum()
# using isna to find out missing values
data.isna().values
data.isna().values.any()
data.isna().sum()
#Find the missing data percentage 
# It looks too much misssing and na values in the data.
#This is required to judge if a column should be taken into consideration for analysis or not in the current form.
#With the shape we already found the value as (239677, 29) but to make it general we need to find the count at runtime.
missing_data_percentage =(data.isnull().sum()/data.shape[0]) * 100
missing_data_percentage
###############################################################################
# https://stackoverflow.com/questions/21925114/is-there-an-implementation-of-
#missingmaps-in-pythons-ecosystem
###############################################################################
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

colours = {True: "#FFD700", False: "#8B0000"}
ax = missmap(data, colors = colours)
plt.show(ax)
###############################################################################
#Analyzing other different columns 
###############################################################################
data['gun_stolen']
data['participant_age_group']
data['participant_age']
data['participant_gender']
data['participant_relationship']
data['participant_status']
data['participant_type']
###############################################################################
# Created a new column for the total number of participants (injured+killed) 
#as per the data available
###############################################################################
data["participants_count"] = data["n_killed"] + data["n_injured"]
###############################################################################
# Create new column for handling the dates
#Converting the date column data type to datatime to apply the date related
#functions
###############################################################################

data["date"] = pd.to_datetime(data["date"])

data["day"] = data["date"].dt.day
data["month"] = data["date"].dt.month
data["year"] = data["date"].dt.year
data["weekday"] = data["date"].dt.weekday
data["week"] = data["date"].dt.week
data["quarter"] = data["date"].dt.quarter
data["weekday"] 
###################################################################################################
# Creating multiple columns from Participant's Gender column
#Fill the na columns with "0::Unknown", to make this similare with existing columns
#This make easier to apply the new crated fundtions 
###################################################################################################
data["participant_gender"] = data["participant_gender"].fillna("0::Unknown")


def participant_gender_formating(row) :
    gender_row_values = []
    gender_row = str(row).split("||")
    for x in gender_row :
        gender_row_value = str(x).split("::")
        if len(gender_row_value) > 1 :
            gender_row_values.append(gender_row_value[1])
            
    return gender_row_values


participant_genders = data.participant_gender.apply(participant_gender_formating)
data["participant_gender_total"] = participant_genders.apply(lambda x: len(x))
data["participant_gender_male"] = participant_genders.apply(lambda x: x.count("Male"))
data["participant_gender_female"] = participant_genders.apply(lambda x: x.count("Female"))
data["participant_gender_unknown"] = participant_genders.apply(lambda x: x.count("Unknown"))

del(participant_genders) #Delete the new created temporary columns

###############################################################################
#Participents age group column holds the data for all the involved persons. This
#data is not much meaningful for analysis. So creating more columns to analyze the
#data proaperly.
###############################################################################

data["participant_age_group"].unique()
data["participant_age_group"] = data["participant_age_group"].fillna("0::Unknown")
data["participant_age_group"]

def participant_age_group_format(row) :
    unknownCount = 0
    adultCount = 0
    teenCount = 0
    childCount = 0
    agegroup_row_values = []
    agegroup_row = str(row).split("||")
    for x in agegroup_row :
            agegroup_row_value = str(x).split("::")
            if len(agegroup_row_value) > 1 :
                agegroup_row_values.append(agegroup_row_value[1])
                if "Adult 18+" in agegroup_row_value :
                    adultCount += 1
                elif "Teen 12-17" in agegroup_row_value :
                    teenCount += 1
                elif "Child 0-11" in agegroup_row_value :
                    childCount += 1
                else :
                    unknownCount += 1
                
    return agegroup_row_values
participantagegroup = data.participant_age_group.apply(participant_age_group_format)
data["adult_participant"] = participantagegroup.apply(lambda x: x.count("Adult 18+"))
data["teen_participant"] = participantagegroup.apply(lambda x: x.count("Teen 12-17"))
data["child_participant"] = participantagegroup.apply(lambda x: x.count("Child 0-11"))
data["unknown_agegroup_participant"] = participantagegroup.apply(lambda x: x.count("Unknown"))

del(participantagegroup)  #Delete the new created temporary column
###############################################################################
#Removing the columns that are not useful for analysis
###############################################################################
data.drop([
    "incident_id",
    "incident_url",
    "sources",
    "source_url",
    "incident_url_fields_missing",
    "location_description",
    "participant_relationship",
    ], axis=1, inplace=True)

###############################################################################  
# Aapply .corr directly to  dataframe, it will return all pairwise correlations 
#between the columns; 
###############################################################################
sns.heatmap(data.corr(), linewidth=.7, cmap="YlGnBu", annot=False, fmt="d")
##############################################################################
# Status of gun violence as per state.
##############################################################################


violence_rate_statemwise = sns.countplot(x=data["state"], data=data, palette=btui, order=data["state"].value_counts().index)
violence_rate_statemwise.set_xticklabels(violence_rate_statemwise.get_xticklabels(), rotation=90)
violence_rate_statemwise.set_title("State(s) with highest number of Gun Violence")
###############################################################################
#Top 10 State having Gun Violence 
###############################################################################
violence_rate_statemwise = data["state"].value_counts().head(10)
violence_rate_statemwise

plt.pie(violence_rate_statemwise,  labels=violence_rate_statemwise.index, colors=btui, autopct="%1.1f%%", shadow=True, startangle=195)
plt.title("State-wise Gun Violence")
plt.axis("equal")
###############################################################################
#Top 10 city affected with gun violence 
###############################################################################
violence_rate_citywise = data["city_or_county"].value_counts().head(10)
violence_rate_citywise

plt.pie(violence_rate_citywise,  labels=violence_rate_citywise.index, colors=btui, autopct="%1.1f%%", shadow=True, startangle=195)
plt.title("City wise Gun Violence")
plt.axis("equal")
###############################################################################
#Barplot :: Top 10 city affected with gun violence 
###############################################################################

violence_rate_top10_citywise = data["city_or_county"].value_counts().head(10)
topcitywise_violence_rate_plot = sns.barplot(x=violence_rate_top10_citywise.index, y=violence_rate_top10_citywise, palette=btui)
topcitywise_violence_rate_plot.set_xticklabels(topcitywise_violence_rate_plot.get_xticklabels(), rotation=50)
topcitywise_violence_rate_plot.set_title("Top 10 Cities or Counties with highest number of Gun Violence")
###############################################################################
#Joinplot :: Lets find in total number of particiapants how many was injured 
#and how many of them killed
###############################################################################


sns.jointplot(x=data.participant_gender_total, y=data.n_killed, data=data, space=0, dropna=True, color="#D81B60")
sns.jointplot(x=data.participant_gender_total, y=data.n_injured, data=data, space=0, dropna=True, color="#1E88E5")

###############################################################################
#Joinplot :: Lets find in total number of particiapants how many was male and female
###############################################################################


sns.jointplot(x=data.participant_gender_total, y=data.participant_gender_male, data=data, space=0, dropna=True, color="#D81B60")
sns.jointplot(x=data.participant_gender_total, y=data.participant_gender_female, data=data, space=0, dropna=True, color="#1E88E5")

###############################################################################
#Barplot :: Showing the killed and injured per year
###############################################################################

fig, axs = plt.subplots(ncols=2)
              
gun_violence_per_year_killed = data.groupby(data["year"]).apply(lambda x: pd.Series(dict(total_killed_per_year = x.n_killed.sum())))                          
gun_violence_per_year_killed_plot = sns.barplot(x=gun_violence_per_year_killed.index, y=gun_violence_per_year_killed.total_killed_per_year, palette=btui, ax=axs[0])
gun_violence_per_year_killed_plot.set_title("Killed each year")

del(gun_violence_per_year_killed)  #Delete the new created temporary column

gun_violence_per_year_injured = data.groupby(data["year"]).apply(lambda x: pd.Series(dict(total_injured_per_year = x.n_injured.sum())))                          
gun_violence_per_year_injured_plot = sns.barplot(x=gun_violence_per_year_injured.index, y=gun_violence_per_year_injured.total_injured_per_year, palette=btui, ax=axs[1])
gun_violence_per_year_injured_plot.set_title("Injured each year")
del(gun_violence_per_year_injured)
###############################################################################
#Barplot :: Showing the killed and injured per month
###############################################################################              
  
fig, axs = plt.subplots(ncols=2)
            
gun_violence_per_month_killed = data.groupby(data["month"]).apply(lambda x: pd.Series(dict(total_killed_per_month = x.n_killed.sum())))                          
gun_violence_per_month_killed_plot = sns.barplot(x = gun_violence_per_month_killed.index , y = gun_violence_per_month_killed.total_killed_per_month ,palette=btui, ax=axs[0]  )
gun_violence_per_month_killed_plot.set_title("Killed per month")
del(gun_violence_per_month_killed)  #Delete the new created temporary column
gun_violence_per_month_injured = data.groupby(data["month"]).apply(lambda x: pd.Series(dict(total_injured_per_month = x.n_injured.sum())))                          
gun_violence_per_month_injured_plot = sns.barplot(x = gun_violence_per_month_injured.index , y= gun_violence_per_month_injured.total_injured_per_month ,palette=btui, ax=axs[1]  )              
gun_violence_per_month_injured_plot.set_title("Injured per month")
del(gun_violence_per_month_injured)  #Delete the new created temporary column

###############################################################################
#Barplot :: Showing the killed and injured per weekday
###############################################################################  
             
fif, axs = plt.subplots(ncols=2)  
gun_violence_per_week_killed = data.groupby(data["weekday"]).apply(lambda x: pd.Series(dict(total_killed_per_week = x.n_killed.sum())))                          
gun_violence_per_week_killed_plot = sns.barplot(x = gun_violence_per_week_killed.index , y = gun_violence_per_week_killed.total_killed_per_week ,palette=btui, ax=axs[0]  )
gun_violence_per_week_killed_plot.set_title("Killed per week") 
del(gun_violence_per_week_killed)  #Delete the new created temporary column
gun_violence_per_week_injured = data.groupby(data["weekday"]).apply(lambda x: pd.Series(dict(total_injured_per_week = x.n_injured.sum())))                          
gun_violence_per_week_injured_plot = sns.barplot(x = gun_violence_per_week_injured.index , y = gun_violence_per_week_injured.total_injured_per_week ,palette=btui, ax=axs[1]  )
gun_violence_per_week_killed_plot.set_title("Injured per week")            
del(gun_violence_per_week_injured)  #Delete the new created temporary column   
###############################################################################
#Boxplot :: Trying to find that in violence which gender was more invloved
############################################################################### 


participant_genders_sum = data[["state", "participant_gender_total", "participant_gender_male", "participant_gender_female", "participant_gender_unknown"]].groupby(data["state"]).sum()
sns.boxplot(data=participant_genders_sum, palette=btui)

###############################################################################
#Density plot :: Involvement in crime as per agegroup.
############################################################################### 

age_group_involvement = data[["adult_participant", "teen_participant","child_participant","unknown_agegroup_participant"]].groupby(data["year"]).sum()
density_plot=sns.kdeplot(age_group_involvement['adult_participant'],shade=True,color="r")
density_plot=sns.kdeplot(age_group_involvement['teen_participant'],shade=True,color="b")
density_plot=sns.kdeplot(age_group_involvement['child_participant'],shade=True,color="g")
density_plot=sns.kdeplot(age_group_involvement['unknown_agegroup_participant'],shade=True,color="y")
del(age_group_involvement)  #Delete the new created temporary column
###############################################################################
#Density tplot :: Gender involvement in the crime
############################################################################### 

gender_participant = data[["participant_gender_total","participant_gender_male", "participant_gender_female"]].groupby(data["year"]).sum()
density_plot=sns.kdeplot(gender_participant['participant_gender_total'],shade=True,color="r")
density_plot=sns.kdeplot(gender_participant['participant_gender_male'],shade=True,color="b")
density_plot=sns.kdeplot(gender_participant['participant_gender_female'],shade=True,color="b")
###############################################################################
#violin plot :: Gender involvement in the crime
###############################################################################

impact_total_gender = data[["participant_gender_total","participant_gender_male","participant_gender_female"]].groupby(data["year"]).sum()
print(impact_total_gender)
impact_total_gender_plot=sns.violinplot(data=impact_total_gender,split=True,inner="quartile")
# SUMMARY:
#    1. The gun violence data collection was from year 2013 to 2018
#    2. In year 2013 the gun violence was very low.
#    3. Sudden increase of gun violence in year 2014. 
#    4. Highest gun violence happend in year 2017.
#    5. Sudden decrease of gun viloence in 2018
#    6. Male members are more involved in crimes.
#    7. More gun violence happened on Sunday.
#    8. Adults are more involved in crimes. 
