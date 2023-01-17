#Reset memory and Call libraries
%reset -f
#Data manipulation modules
import pandas as pd 
import numpy as np
#For pltting
import matplotlib.pyplot as plt
import seaborn as sns    
#Misc
import os
#Show graphs in a separate window
%matplotlib inline

#Read data file
gun_violence_data = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")

gun_violence_data.head()
gun_violence_data.shape
gun_violence_data.columns
# using isnull to find out missing values
gun_violence_data.isnull().values.any()

gun_violence_data.isnull().sum()
gun_violence_data.dtypes
gun_violence_data.describe()
gun_violence_data.info()
gun_violence_data.drop([
    "incident_id",
    "incident_url",
    "sources",
    "source_url",
    "incident_url_fields_missing",
    "location_description",
    "participant_relationship",
    ], axis=1, inplace=True)

##Converting object datatype to datetime
gun_violence_data["date"] = pd.to_datetime(gun_violence_data["date"])

## To create column-day,month,year,weekday ,week and quarter.
gun_violence_data["day"] = gun_violence_data["date"].dt.day
gun_violence_data["month"] = gun_violence_data["date"].dt.month
gun_violence_data["year"] = gun_violence_data["date"].dt.year
gun_violence_data["weekday"] = gun_violence_data["date"].dt.weekday
gun_violence_data["week"] = gun_violence_data["date"].dt.week
gun_violence_data["quarter"] = gun_violence_data["date"].dt.quarter
gun_violence_data.isnull().values.any()
gun_violence_data.isnull().sum()
gun_violence_data.isna().values.any()
missingdata_sum=gun_violence_data.isna().sum()
missingdata_count=gun_violence_data.isna().count()
percentage_missingdata=(missingdata_sum/missingdata_count) * 100
missingdata = pd.concat([missingdata_sum, percentage_missingdata], axis=1)
missingdata
# Created a new column for the total number of persons impacted (injured+killed) as per the data available
gun_violence_data["total_impacted"] = gun_violence_data["n_killed"] + gun_violence_data["n_injured"]

# Creating multiple columns from Participant's Gender column
gun_violence_data["participant_gender"] = gun_violence_data["participant_gender"].fillna("0::Unknown")


def clean_participant_gender(row) :
    gender_row_values = []
    gender_row = str(row).split("||")
    for x in gender_row :
        gender_row_value = str(x).split("::")
        if len(gender_row_value) > 1 :
            gender_row_values.append(gender_row_value[1])
            
    return gender_row_values


participant_genders = gun_violence_data.participant_gender.apply(clean_participant_gender)
gun_violence_data["participant_gender_total"] = participant_genders.apply(lambda x: len(x))
gun_violence_data["participant_gender_male"] = participant_genders.apply(lambda x: x.count("Male"))
gun_violence_data["participant_gender_female"] = participant_genders.apply(lambda x: x.count("Female"))
gun_violence_data["participant_gender_unknown"] = participant_genders.apply(lambda x: x.count("Unknown"))
del(participant_genders)

# Checking for null value of column for guns involved and guns stolen and filling the missing values
gun_violence_data["n_guns_involved"] = gun_violence_data["n_guns_involved"].fillna(value =0)
gun_violence_data["gun_stolen"] = gun_violence_data["gun_stolen"].fillna(value = "0::Unknown")
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

gunstolenvalues = gun_violence_data.gun_stolen.apply(clean_gun_stolen)
gun_violence_data["gun_stolen_stolen"] = gunstolenvalues.apply(lambda x: x.count("Stolen"))
gun_violence_data["gun_stolen_notstolen"] = gunstolenvalues.apply(lambda x: x.count("Not-stolen"))
del(gunstolenvalues)
# Checking values for new columns added
gun_violence_data.head()
#Joint Distribution plots:

#To draw plot for number of guns involved vs guns stolen.
sns.jointplot(
         "n_guns_involved", 
         "gun_stolen_stolen",
         gun_violence_data,
         kind="scatter",color="#FF0000")
#To draw plot for number of participant gender total vs number of killed.
sns.jointplot(
         "participant_gender_total", 
         "n_killed",
         gun_violence_data,
         kind="scatter", color="#D81B60")
#To draw plot for number of participant gender total vs number of injured
sns.jointplot(
         "participant_gender_total", 
         "n_injured",
         gun_violence_data,
         kind="scatter", color="#1E88E5")
# Jointplot between Number of Person Killed Vs Injured in all incidences
sns.jointplot("n_injured",
              "n_killed",
              gun_violence_data,
              kind='scatter',      # kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }, optional
              s=200, color='m', edgecolor="skyblue", linewidth=2)

#Histograms
crime_rate_state = gun_violence_data["state"].value_counts()

crime_rate_state_plot = sns.countplot(x=gun_violence_data["state"], 
                                 data=gun_violence_data,
                                 order=gun_violence_data["state"].value_counts().index)
crime_rate_state_plot.set_xticklabels(crime_rate_state_plot.get_xticklabels(), rotation=90)
crime_rate_state_plot.set_title("State(s) with highest number of Gun Violence")
crime_rate_city = gun_violence_data["city_or_county"].value_counts().head(20)
crime_rate_city_plot = sns.barplot(x=crime_rate_city.index, y=crime_rate_city)
crime_rate_city_plot.set_xticklabels(crime_rate_city_plot.get_xticklabels(), rotation=75)
crime_rate_city_plot.set_title("Cities or Counties with highest number of Gun Violence")

#PIE Graphs
plt.pie(crime_rate_state, labels=crime_rate_state.index, autopct="%1.1f%%")
plt.title("State-wise Gun Violence Percentage")
plt.axis("equal")
plt.pie(crime_rate_city, labels=crime_rate_city.index, autopct="%1.1f%%")
plt.title("Top 20 City-wise Gun Violence Percentage")
plt.axis("equal")
# Kernel Density plots

# Density plot for genderwise participant
genderwise_total = gun_violence_data[["participant_gender_total", "participant_gender_male", "participant_gender_female", "participant_gender_unknown"]].groupby(gun_violence_data["year"]).sum()
dp_gen_plot=sns.kdeplot(genderwise_total["participant_gender_male"], shade=True, color="r")
dp_gen_plot=sns.kdeplot(genderwise_total["participant_gender_female"], shade=True, color="b")
dp_gen_plot=sns.kdeplot(genderwise_total['participant_gender_unknown'], shade=True, color="g")
del(genderwise_total)
# Density plot for person injured vs killed on all weekdays
inj_kill_weektotal = gun_violence_data[["n_injured","n_killed"]].groupby(gun_violence_data["weekday"]).sum()
dp_inj_kill_plot=sns.kdeplot(inj_kill_weektotal['n_injured'], shade=True, color="r")
dp_inj_kill_plot=sns.kdeplot(inj_kill_weektotal['n_killed'], shade=True, color="b")
del(inj_kill_weektotal)
#Point plots
# Point plot showing yearly no of persons Killed 
yearly_vs_killed=gun_violence_data.groupby(gun_violence_data["year"]).apply(lambda x: pd.Series(dict(No_Killed=x.n_killed.sum())))
yearly_vs_killed_plot=sns.pointplot(x=yearly_vs_killed.index, y=yearly_vs_killed.No_Killed, data=yearly_vs_killed,label="yearly_vs_killed")
yearly_vs_killed
# Point plot showing yearly no of persons Injured
yearly_vs_injured=gun_violence_data.groupby(gun_violence_data["year"]).apply(lambda x: pd.Series(dict(No_Injured=x.n_injured.sum())))
yearly_vs_injured_plot=sns.pointplot(x=yearly_vs_injured.index, y=yearly_vs_injured.No_Injured, data=yearly_vs_injured,label="yearly_vs_injured")
yearly_vs_injured
#Violin plot
# Violin Plot for Year wise no of people injured
year_vs_injured_plot = sns.violinplot("year", "n_injured", data=gun_violence_data,split=True, inner="quartile")
year_vs_injured_plot.set_title("Persons injured in the incidents per Year")

# Violin Plot for Year wise no of people killed
year_vs_killed_plot = sns.violinplot("year", "n_killed",
               data=gun_violence_data,
               split=True,         # If hue variable has two levels, draw half of a violin for each level.
               inner="quartile"    #  Options: “box”, “quartile”, “point”, “stick”, None 
               )
year_vs_killed_plot.set_title("Persons killed in the incidents per Year")
#Box plot
# Box plot for total number of persons killed State wise
sns.boxplot('state','n_killed',data=gun_violence_data)
#Facet grid graph

# Facet Grid Graph for Male/ Female Partipant per Year
g = sns.FacetGrid(gun_violence_data, hue="year", palette="Set1", size=5, hue_kws={"marker": ["^", "v","*",">","<","."]})
g.map(plt.scatter, "participant_gender_male", "participant_gender_female", s=100, linewidth=.5, edgecolor="white")
g.add_legend();