# Project: Gun Violence Data Exploring, Analysis & Plot Model

# Data Preparation, Feature Engineering, and Exploratory Analysis Gun Violence Archive (GVA) is a not for profit corporation formed in 2013
# to provide free online public access to accurate information about gun-related 
# violence in the United States. GVA will collect and check for accuracy, comprehensive 
# information about gun-related violence in the U.S. and then post and disseminate it online.

# DataSet: gun-violence-data_01-2013_03-2018.csv (142.76 MB) - Gun Violence - 
# This dataset contains information incident_id - ID of the crime report
#•date - Date of crime
#•state - State of crime
#•city_or_county - City/ County of crime
#•address - Address of the location of the crime
#•n_killed - Number of people killed
#•n_injured - Number of people injured
#•incident_url - URL regarding the incident
#•source_url - Reference to the reporting source
#•incident_url_fields_missing - TRUE if the incident_url is present, FALSE otherwise
#•congressional_district - Congressional district id
#•gun_stolen - Status of guns involved in the crime (i.e. Unknown, Stolen, etc...)
#•gun_type - Typification of guns used in the crime
#•incident_characteristics - Characteristics of the incidence
#•latitude - Location of the incident
#•location_description
#•longitude - Location of the incident
#•n_guns_involved - Number of guns involved in incident
#•notes - Additional information of the crime
#•participant_age - Age of participant(s) at the time of crime
#•participant_age_group - Age group of participant(s) at the time crime
#•participant_gender - Gender of participant(s)
#•participant_name - Name of participant(s) involved in crime
#•participant_relationship - Relationship of participant to other participant(s)
#•participant_status - Extent of harm done to the participant
#•participant_type - Type of participant
#•sources
#•state_house_district
#•state_senate_district

#Graphs to be plotted here:
#i)  Joint Distribution plots
#ii)  Histograms
#iii) Kernel Density plots
#iv) Violin plots
#v) Box plots
#vi) FacetGrid
#vii) Pie Graphs
#Lets import some libraries

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#To show graph in same window.
%matplotlib inline
# Read data file
data_gunv = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")

# Explore data records of Gun Violance data
data_gunv.head() 
data_gunv.columns
data_gunv.dtypes
data_gunv.shape
data_gunv.describe()
data_gunv.info()
data_gunv['gun_type'].unique()
# Converting object datatype to datetime
data_gunv['date'] = pd.to_datetime(data_gunv['date']) 
# We can create columns for weekdays, month, year and extract values 
# from date for further analysis
data_gunv['f_weekday'] = data_gunv['date'].dt.weekday
data_gunv['f_month'] = data_gunv['date'].dt.month
data_gunv['f_year'] = data_gunv['date'].dt.year
data_gunv['f_weekday'] = data_gunv['f_weekday'].astype('object')
data_gunv['f_year'] = data_gunv['f_year'].astype('object')
data_gunv['f_month'] = data_gunv['f_month'].astype('object')

#Check the datatype of columns are changed
data_gunv.dtypes
# Checking for null value of column for guns involved and guns stolen 
data_gunv["n_guns_involved"] = data_gunv["n_guns_involved"].fillna(value =0)
data_gunv["gun_stolen"] = data_gunv["gun_stolen"].fillna(value = "0::Unknown")
# Created column for total number of persons impacted (injured+killed)
data_gunv['total_impacted'] = data_gunv['n_killed'] + data_gunv['n_injured']
# Creating multiple columns from Participant's Gender column
data_gunv["participant_gender"] = data_gunv["participant_gender"].fillna("0::Unknown")
    
def gen(n) :                    
    gen_rows = []               
    gen_row = str(n).split("||")    
    for i in gen_row :              
        g_row = str(i).split("::")  
        if len(g_row) > 1 :         
            gen_rows.append(g_row[1])    

    return gen_rows

gen_series = data_gunv.participant_gender.apply(gen)
data_gunv["total_participant"] = gen_series.apply(lambda x: len(x))
data_gunv["male_participant"] = gen_series.apply(lambda i: i.count("Male"))
data_gunv["female_participant"] = gen_series.apply(lambda i: i.count("Female"))
data_gunv["unknown_participant"] = gen_series.apply(lambda i: i.count("Unknown"))
# Checking values for new columns
data_gunv.head()
data_gunv.shape
#Now let us try to plot below mentioned graphs:
#i)  Joint Distribution plots
#ii)  Histograms
#iii) Kernel Density plots
#iv) Violin plots
#v) Box plots
#vi) FacetGrid
# Draw a jointplot between Number of Person Killed Vs Injured in all incidences
sns.jointplot("n_injured",
              "n_killed",
              data_gunv,
              kind='scatter',      # kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }, optional
              s=200, color='m', edgecolor="skyblue", linewidth=2)
# Draw a jointplot to identify Maximum Number of Person Injured in which incidence
sns.jointplot("incident_id",
              "n_injured",
              data_gunv,
              kind='hex'      # kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }, optional
              )
# Draw a jointplot to identify Maximum Number of Person Killed in which incidence
sns.jointplot("incident_id",
              "n_killed",
              data_gunv,
              kind='scatter',      # kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }, optional
              color="Red",
              marginal_kws={'color': 'blue'})
# Plot a Histogram for Top 10 Cities with maximum incidents of Gun Violence
ctwise_total = data_gunv[["incident_id"]].groupby(data_gunv["city_or_county"]).count()
top_ct = ctwise_total.sort_values(by='incident_id', ascending=False).head(10)
print(top_ct)
top_ct.plot.barh()
del(top_ct)
# Plot a Histogram for Weekday wise Incidents
weekwise_total = data_gunv[["incident_id"]].groupby(data_gunv["f_weekday"]).count()
weekwise_total.plot.barh()
del(weekwise_total)
# Here, for weekdays Monday is 0 and Sunday is 6.
# Plot a Histogram for Top 10 States with maximum incidents of Gun Violence
stwise_total = data_gunv[["incident_id"]].groupby(data_gunv["state"]).count()
top_st = stwise_total.sort_values(by='incident_id', ascending=False).head(10)
print(top_st)
top_st.plot.barh()
del(top_st)
# Density plot for gendrwise participant
genderwise_total = data_gunv[["total_participant", "male_participant", "female_participant", "unknown_participant"]].groupby(data_gunv["f_year"]).sum()
dp_gen_plot=sns.kdeplot(genderwise_total['male_participant'], shade=True, color="r")
dp_gen_plot=sns.kdeplot(genderwise_total['female_participant'], shade=True, color="b")
dp_gen_plot=sns.kdeplot(genderwise_total['unknown_participant'], shade=True, color="g")
del(genderwise_total)
# Density plot for person injured vs killed on all weekdays
inj_kill_weektotal = data_gunv[["n_injured","n_killed"]].groupby(data_gunv["f_weekday"]).sum()
dp_inj_kill_plot=sns.kdeplot(inj_kill_weektotal['n_injured'], shade=True, color="r")
dp_inj_kill_plot=sns.kdeplot(inj_kill_weektotal['n_killed'], shade=True, color="b")
del(inj_kill_weektotal)
# Violin Plot for Yearwise Person Injured
yr_injured_plot = sns.violinplot("f_year", "n_injured", data=data_gunv,
                                 split=True, inner="quartile")
yr_injured_plot.set_title("Person killed in incidents per Year")
#  Violin Plot for  Yearwise Person killed
yr_killed_plot = sns.violinplot("f_year", "n_killed",
               data=data_gunv,
               split=True,         # If hue variable has two levels, draw half of a violin for each level.
               inner="quartile"    #  Options: “box”, “quartile”, “point”, “stick”, None 
               )
#Violin Plot for Peron Impacted(Killed/Injured) during gun violence
Impacted_person_total = data_gunv[["total_impacted", "n_injured", "n_killed"]].groupby(data_gunv["f_year"]).sum()
print(Impacted_person_total)
yr_impacted_plot = sns.violinplot(data=Impacted_person_total,
               split=True,         # If hue variable has two levels, draw half of a violin for each level.
               inner="quartile"    #  Options: “box”, “quartile”, “point”, “stick”, None 
               )
del(Impacted_person_total)
#Violin Plot for Genderwise Peron involved/impacted during gun violence
genderwise_total = data_gunv[["total_participant", "male_participant", "female_participant", "unknown_participant"]].groupby(data_gunv["f_year"]).sum()
print(genderwise_total)
yr_gender_plot = sns.violinplot(data=genderwise_total,
               split=True,         # If hue variable has two levels, draw half of a violin for each level.
               inner="quartile"    #  Options: “box”, “quartile”, “point”, “stick”, None 
               )
del(genderwise_total)
# Box Plot for Monthwise Person Injured
mth_injured_plot = sns.boxplot("f_month", "n_injured", data= data_gunv)
mth_injured_plot.set_title("Person injured in incidents per month")
# Box Plot for Monthwise Person Killed
mth_killed_plot = sns.boxplot("f_month", "n_killed", data= data_gunv)
mth_killed_plot.set_title("Person killed in incidents per month")
# Count Plot for Statewise incidences of Gun Violence
state_inc_plot = sns.countplot("state", data = data_gunv)
state_inc_plot.set_title("State wise incidence of Gun Violence")
state_inc_plot.set_xticklabels(state_inc_plot.get_xticklabels(), rotation=90)
statewise_crime_rate = data_gunv["state"].value_counts()
statewise_crime_rate
plt.pie(statewise_crime_rate, labels=statewise_crime_rate.index, autopct="%1.1f%%", shadow=True, startangle=195)
plt.title("Gun Violence Percentage statewise")
plt.axis("equal")
topcitywise_crime_rate = data_gunv["city_or_county"].value_counts().head(50)
plt.pie(topcitywise_crime_rate, labels=topcitywise_crime_rate.index, autopct="%1.1f%%", shadow=True, startangle=195)
plt.title("Gun Violence Percentage Citywise")
plt.axis("equal")
# Count Plot for State House District wise
state_inc_plot = sns.countplot("state_house_district", data = data_gunv)
state_inc_plot.set_title("State House Districtwise incidence of Gun Violence")
state_inc_plot.set_xticklabels(state_inc_plot.get_xticklabels())

# Count Plot for State Senate District wise
state_inc_plot = sns.countplot("state_senate_district", data = data_gunv)
state_inc_plot.set_title("State Senate District wise incidence of Gun Violence")
state_inc_plot.set_xticklabels(state_inc_plot.get_xticklabels())
# Count Plot for Weekwise incidences of Gun Violence
wk_inc_plot = sns.countplot("f_weekday", data = data_gunv)
wk_inc_plot.set_title("Weekwise incidence of Gun Violence")
# Count Plot for Monthwise incidences of Gun Violence
mth_inc_plot = sns.countplot("f_month", data = data_gunv)
mth_inc_plot.set_title("Monthwise incidence of Gun Violence")
# Count Plot for Yearwise incidences of Gun Violence
yr_inc_plot = sns.countplot("f_year", data = data_gunv)
yr_inc_plot.set_title("Yearwise incidence of Gun Violence")

# Facet Grid Graph for Male/ Female Partipant per Year
g = sns.FacetGrid(data_gunv, hue="f_year", palette="Set1", size=5, hue_kws={"marker": ["^", "v","*",">","<","o"]})
g.map(plt.scatter, "male_participant", "female_participant", s=100, linewidth=2, edgecolor="black")
g.add_legend();
# Facet Grid Graphh for Person killed and Injured per Year
g = sns.FacetGrid(data_gunv, hue="f_year", palette="Set1", size=5, hue_kws={"marker": ["^", "v","*",">","<","o"]})
g.map(plt.scatter, "n_injured", "n_killed", s=100, linewidth=2, edgecolor="red")
g.add_legend();
# Facet Grid Graphh for Person killed and Injured on Particular days of the week
g = sns.FacetGrid(data_gunv, hue="f_weekday", palette="Set1", size=5, hue_kws={"marker": ["^", "v","h","o",">","<","d"]})
g.map(plt.scatter, "n_injured", "n_killed", s=100, linewidth=.5, edgecolor="yellow")
g.add_legend();
#According to above analysis put forth, We can be able to make predictions on GVD