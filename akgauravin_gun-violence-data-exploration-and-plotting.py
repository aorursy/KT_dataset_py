# Data manipulation modules
import pandas as pd        # R-like data manipulation
import numpy as np         # n-dimensional arrays

# For plotting
import matplotlib as mpl
import matplotlib.pyplot as plt      # For base plotting
# Seaborn is a library for making statistical graphics
# in Python. It is built on top of matplotlib and 
#  numpy and pandas data structures.
import seaborn as sns                # Easier plotting

# Misc
import os

## To Show graphs in same window
%matplotlib inline

mpl.style.use("seaborn")
plt.style.use("seaborn")

######### Begin
# Read data file
data_gv = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")

# Explore data - First 5 records of Gun Violance data
data_gv.head()                          # head()


data_gv.columns
data_gv.columns.values
data_gv.dtypes
data_gv.describe()
data_gv.info()
data_gv.shape
# Removing columns not useful in analysis
data_gv.drop(["incident_characteristics",
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
data_gv['gun_type'].unique()              # Which values
##Converting object datatype to datetime
data_gv['date'] = pd.to_datetime(data_gv['date']) 
# we can create columns for month, year and weekdays and extract values 
# from date for further analysis
data_gv['f_month'] = data_gv['date'].dt.month
data_gv['f_year'] = data_gv['date'].dt.year
data_gv['f_weekday'] = data_gv['date'].dt.weekday

data_gv['f_year'] = data_gv['f_year'].astype('object')
data_gv['f_month'] = data_gv['f_month'].astype('object')
data_gv['f_weekday'] = data_gv['f_weekday'].astype('object')

#Check the datatype of columns are changed
data_gv.dtypes
# Created column for total number of persons impacted (injured+killed)
data_gv['total_impacted'] = data_gv['n_killed'] + data_gv['n_injured']
# Checking for null value of column for guns involved and guns stolen 
data_gv["n_guns_involved"] = data_gv["n_guns_involved"].fillna(value =0)
data_gv["gun_stolen"] = data_gv["gun_stolen"].fillna(value = "0::Unknown")
## Creating multiple columns from Participant's Gender column
data_gv["participant_gender"] = data_gv["participant_gender"].fillna("0::Unknown")
    
def gen(n) :                    
    gen_rows = []               
    gen_row = str(n).split("||")    
    for i in gen_row :              
        g_row = str(i).split("::")  
        if len(g_row) > 1 :         
            gen_rows.append(g_row[1])    

    return gen_rows

gen_series = data_gv.participant_gender.apply(gen)
data_gv["total_participant"] = gen_series.apply(lambda x: len(x))
data_gv["male_participant"] = gen_series.apply(lambda i: i.count("Male"))
data_gv["female_participant"] = gen_series.apply(lambda i: i.count("Female"))
data_gv["unknown_participant"] = gen_series.apply(lambda i: i.count("Unknown"))

# Checking values for new columns
data_gv.head()

data_gv.shape

##As per assignment lets plot following graphs:
#i)  Joint Distribution plots
#ii)  Histograms
#iii) Kernel Density plots
#iv) Violin plots
#v) Box plots
#vi) FacetGrid
###########################Joint Distribution plots############################
# Draw a jointplot between Number of Person Killed Vs Injured in all incidences
sns.jointplot("n_injured",
              "n_killed",
              data_gv,
              kind='scatter',      # kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }, optional
              s=200, color='m', edgecolor="skyblue", linewidth=2)

# Draw a jointplot to identify Maximum Number of Person Injured in which incidence
sns.jointplot("incident_id",
              "n_injured",
              data_gv,
              kind='scatter'      # kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }, optional
              )
# Draw a jointplot to identify Maximum Number of Person Killed in which incidence
sns.jointplot("incident_id",
              "n_killed",
              data_gv,
              kind='scatter',      # kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }, optional
              color="Red",
              marginal_kws={'color': 'red'})
###############################  Histograms  #########################

# Plot a Histogram for Top 10 Cities with maximum incidents of Gun Violence
ctwise_total = data_gv[["incident_id"]].groupby(data_gv["city_or_county"]).count()
top_ct = ctwise_total.sort_values(by='incident_id', ascending=False).head(10)
print(top_ct)
top_ct.plot.barh()
del(top_ct)
# Plot a Histogram for Top 10 States with maximum incidents of Gun Violence
stwise_total = data_gv[["incident_id"]].groupby(data_gv["state"]).count()
top_st = stwise_total.sort_values(by='incident_id', ascending=False).head(10)
print(top_st)
top_st.plot.barh()
del(top_st)

# Plot a Histogram for Weekday wise Incidents
weekwise_total = data_gv[["incident_id"]].groupby(data_gv["f_weekday"]).count()
weekwise_total.plot.barh()
del(weekwise_total)
# Here, for weekdays Monday is 0 and Sunday is 6.
############################  Kernel Density plots  #################################
# Density plot for gendrwise participant
genderwise_total = data_gv[["total_participant", "male_participant", "female_participant", "unknown_participant"]].groupby(data_gv["f_year"]).sum()
dp_gen_plot=sns.kdeplot(genderwise_total['male_participant'], shade=True, color="r")
dp_gen_plot=sns.kdeplot(genderwise_total['female_participant'], shade=True, color="b")
dp_gen_plot=sns.kdeplot(genderwise_total['unknown_participant'], shade=True, color="g")
del(genderwise_total)
# Density plot for person injured vs killed on all weekdays
inj_kill_weektotal = data_gv[["n_injured","n_killed"]].groupby(data_gv["f_weekday"]).sum()
dp_inj_kill_plot=sns.kdeplot(inj_kill_weektotal['n_injured'], shade=True, color="r")
dp_inj_kill_plot=sns.kdeplot(inj_kill_weektotal['n_killed'], shade=True, color="b")
del(inj_kill_weektotal)
################################## Violin plots #################################
# Violin Plot for Yearwise Person Injured
yr_injured_plot = sns.violinplot("f_year", "n_injured", data=data_gv,
                                 split=True, inner="quartile")
yr_injured_plot.set_title("Person killed in incidents per Year")
#  Violin Plot for  Yearwise Person killed
yr_killed_plot = sns.violinplot("f_year", "n_killed",
               data=data_gv,
               split=True,         # If hue variable has two levels, draw half of a violin for each level.
               inner="quartile"    #  Options: “box”, “quartile”, “point”, “stick”, None 
               )

#Violin Plot for Peron Impacted(Killed/Injured) during gun violence
Impacted_person_total = data_gv[["total_impacted", "n_injured", "n_killed"]].groupby(data_gv["f_year"]).sum()
print(Impacted_person_total)
yr_impacted_plot = sns.violinplot(data=Impacted_person_total,
               split=True,         # If hue variable has two levels, draw half of a violin for each level.
               inner="quartile"    #  Options: “box”, “quartile”, “point”, “stick”, None 
               )
del(Impacted_person_total)
#Violin Plot for Genderwise Peron involved/impacted during gun violence
genderwise_total = data_gv[["total_participant", "male_participant", "female_participant", "unknown_participant"]].groupby(data_gv["f_year"]).sum()
print(genderwise_total)
yr_gender_plot = sns.violinplot(data=genderwise_total,
               split=True,         # If hue variable has two levels, draw half of a violin for each level.
               inner="quartile"    #  Options: “box”, “quartile”, “point”, “stick”, None 
               )

del(genderwise_total)
###################################  Box plots ##################################
# Box Plot for Monthwise Person Killed
mth_killed_plot = sns.boxplot("f_month", "n_killed", data= data_gv)
mth_killed_plot.set_title("Person killed in incidents per month")

# Box Plot for Monthwise Person Injured
mth_injured_plot = sns.boxplot("f_month", "n_injured", data= data_gv)
mth_injured_plot.set_title("Person injured in incidents per month")

####################################### Count Plot #################################
# Count Plot for Statewise incidences of Gun Violence
state_inc_plot = sns.countplot("state", data = data_gv)
state_inc_plot.set_title("Staterwise incidence of Gun Violence")
state_inc_plot.set_xticklabels(state_inc_plot.get_xticklabels(), rotation=90)
## Count Plot for State House District wise
state_inc_plot = sns.countplot("state_house_district", data = data_gv)
state_inc_plot.set_title("State House District wise incidence of Gun Violence")
state_inc_plot.set_xticklabels(state_inc_plot.get_xticklabels())
# Count Plot for State Senate District wise
state_inc_plot = sns.countplot("state_senate_district", data = data_gv)
state_inc_plot.set_title("State Senate District wise incidence of Gun Violence")
state_inc_plot.set_xticklabels(state_inc_plot.get_xticklabels())

# Count Plot for Weekwise incidences of Gun Violence
wk_inc_plot = sns.countplot("f_weekday", data = data_gv)
wk_inc_plot.set_title("Weekwise incidence of Gun Violence")
# Count Plot for Monthwise incidences of Gun Violence
mth_inc_plot = sns.countplot("f_month", data = data_gv)
mth_inc_plot.set_title("Monthwise incidence of Gun Violence")
# Count Plot for Yearwise incidences of Gun Violence
yr_inc_plot = sns.countplot("f_year", data = data_gv)
yr_inc_plot.set_title("Yearwise incidence of Gun Violence")
################################# FacetGrid ################################
# Facet Grid Graph for Male/ Female Partipant per Year
g = sns.FacetGrid(data_gv, hue="f_year", palette="Set1", size=5, hue_kws={"marker": ["^", "v","*",">","<","o"]})
g.map(plt.scatter, "male_participant", "female_participant", s=100, linewidth=.5, edgecolor="white")
g.add_legend();
# Facet Grid Graphh for Person killed and Injured per Year
g = sns.FacetGrid(data_gv, hue="f_year", palette="Set1", size=5, hue_kws={"marker": ["^", "v","*",">","<","o"]})
g.map(plt.scatter, "n_injured", "n_killed", s=100, linewidth=.5, edgecolor="white")
g.add_legend();

# Facet Grid Graphh for Person killed and Injured on Particular days of the week
g = sns.FacetGrid(data_gv, hue="f_weekday", palette="Set1", size=5, hue_kws={"marker": ["^", "v","h","o",">","<","d"]})
g.map(plt.scatter, "n_injured", "n_killed", s=100, linewidth=.5, edgecolor="white")
g.add_legend();
## Please UPVOTE, if you Like the Data Exploration and Plotting