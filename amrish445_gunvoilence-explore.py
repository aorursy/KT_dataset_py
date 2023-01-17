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
# Load the dataset
GunVoilence_data = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")

# Explore data - First 5 records of Gun Violance data
GunVoilence_data.head()  
#explore data
GunVoilence_data.shape        # 239677 X 29

GunVoilence_data.columns
GunVoilence_data.dtypes 

#To describe the data-gun voilence.
GunVoilence_data.describe

# After executions found that isnull and isna gives same counts/values

GunVoilence_data.isnull().values.any()
GunVoilence_data.isnull().sum()
GunVoilence_data.isna().values.any()

missingdata_sum=GunVoilence_data.isna().sum()
missingdata_count=GunVoilence_data.isna().count()
percentage_missingdata=(missingdata_sum/missingdata_count) * 100
missingdata = pd.concat([missingdata_sum, percentage_missingdata], axis=1)
missingdata
del(missingdata_sum, missingdata_count, percentage_missingdata)
# Removing columns for not using of analysis and plotting
GunVoilence_data.drop(["incident_characteristics",
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
GunVoilence_data.shape
##Converting object datatype to datetime
GunVoilence_data["date"] = pd.to_datetime(GunVoilence_data["date"])


## To create column-day,month,year,weekday ,week and quarter.
GunVoilence_data["day"] = GunVoilence_data["date"].dt.day
GunVoilence_data["month"] = GunVoilence_data["date"].dt.month
GunVoilence_data["year"] = GunVoilence_data["date"].dt.year
GunVoilence_data["weekday"] = GunVoilence_data["date"].dt.weekday
GunVoilence_data["week"] = GunVoilence_data["date"].dt.week
GunVoilence_data["quarter"] = GunVoilence_data["date"].dt.quarter
GunVoilence_data.dtypes # data types creatd for day,month,year,weekday,week and quarter.
GunVoilence_data["participant_gender"] = GunVoilence_data["participant_gender"].fillna("0::Unknown")
GunVoilence_data.n_guns_involved
def clean_participant_gender(row) :
    gender_row_values = []
    gender_row = str(row).split("||")
    for x in gender_row :
        gender_row_value = str(x).split("::")
        if len(gender_row_value) > 1 :
            gender_row_values.append(gender_row_value[1])
            
    return gender_row_values


participant_genders = GunVoilence_data.participant_gender.apply(clean_participant_gender)
GunVoilence_data["participant_gender_total"] = participant_genders.apply(lambda x: len(x))
GunVoilence_data["participant_gender_male"] = participant_genders.apply(lambda x: x.count("Male"))
GunVoilence_data["participant_gender_female"] = participant_genders.apply(lambda x: x.count("Female"))
GunVoilence_data["participant_gender_unknown"] = participant_genders.apply(lambda x: x.count("Unknown"))
del(participant_genders)
GunVoilence_data["n_guns_involved"] = GunVoilence_data["n_guns_involved"].fillna(0)
GunVoilence_data["gun_stolen"] = GunVoilence_data["gun_stolen"].fillna("0::Unknown")
# Prints a lot but gives all the unique values of a column
#dataset_gunviolence["gun_stolen"].unique()

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


gunstolenvalues = GunVoilence_data.gun_stolen.apply(clean_gun_stolen)
GunVoilence_data["gun_stolen"] = gunstolenvalues.apply(lambda x: x.count("Stolen"))
GunVoilence_data["gun_stolen_notstolen"] = gunstolenvalues.apply(lambda x: x.count("Not-stolen"))
del(gunstolenvalues)

###########################Joint Distribution plots############################
#To draw plot for number of guns involved vs guns stolen.
sns.jointplot(x=GunVoilence_data.n_guns_involved, y=GunVoilence_data["gun_stolen"], kind="scatter",color="#FF0000")
#To draw plot for number of guns involved vs guns not stolen.
sns.jointplot(x=GunVoilence_data.n_guns_involved, y=GunVoilence_data["gun_stolen_notstolen"], kind="scatter")

#To draw plot for number of participant gender total vs number of killed.
#To draw plot for number of participant gender total vs number of injured.
sns.jointplot(x=GunVoilence_data.participant_gender_total,
              y=GunVoilence_data.n_killed, data=GunVoilence_data,
              space=0, dropna=True, color="#D81B60")

sns.jointplot(x=GunVoilence_data.participant_gender_total,
              y=GunVoilence_data.n_injured, data=GunVoilence_data,
              space=0, dropna=True, color="#D81B60")
###############################  Histograms  #########################
# Plot a Histogram for Top 10 Cities with maximum incidents of Gun Violence

#for Top 10 states with maximum incidents of Gun Violence

ct_state_total = GunVoilence_data[["incident_id"]].groupby(GunVoilence_data.state).count()
top_ct_state = ct_state_total.sort_values(by='incident_id', ascending=False).head(10)
print(top_ct_state)
#top_ct_state.plot.barh()
#statevise
g = sns.distplot(ct_state_total,bins=50, kde=False,rug=True);
#g.axvline(0, color="red", linestyle="--");
del(top_ct_state)

#countoryvise
ct_country_total = GunVoilence_data[["incident_id"]].groupby(GunVoilence_data.city_or_county).count()
top_ct_country = ct_country_total.sort_values(by='incident_id', ascending=False).head(10)
print(top_ct_country)
g1 = sns.distplot(ct_country_total, kde=False,rug=True,axlabel="Country");
top_ct_country.plot.barh()
del(top_ct_country)
#################################Kernal Density plot #######################################
#To draw plot for yearly incident,injured and killed 
yearly_impact = GunVoilence_data[["n_killed", "n_injured","incident_id"]].groupby(GunVoilence_data["year"]).sum()
density_plot=sns.kdeplot(yearly_impact['n_killed'],shade=True,color="r")
density_plot=sns.kdeplot(yearly_impact['n_injured'],shade=True,color="b")
#density_plot=sns.kdeplot(yearly_impact['incident_id'],shade=True,color="y")
print(yearly_impact['n_killed'])
sns.distplot(yearly_impact['n_killed'], hist=False, rug=True);

#del(yearly_impact)
## Creating multiple columns from Participant's Gender column
GunVoilence_data["participant_gender"] = GunVoilence_data["participant_gender"].fillna("0::Unknown")
    
def gen(n) :                    
    gen_rows = []               
    gen_row = str(n).split("||")    
    for i in gen_row :              
        g_row = str(i).split("::")  
        if len(g_row) > 1 :         
            gen_rows.append(g_row[1])    

    return gen_rows

gen_series = GunVoilence_data.participant_gender.apply(gen)
GunVoilence_data["total_participant"] = gen_series.apply(lambda x: len(x))
GunVoilence_data["male_participant"] = gen_series.apply(lambda i: i.count("Male"))
GunVoilence_data["female_participant"] = gen_series.apply(lambda i: i.count("Female"))
GunVoilence_data["unknown_participant"] = gen_series.apply(lambda i: i.count("Unknown"))
# Density plot for gendrwise participant
genderwise_total = GunVoilence_data[["total_participant", "male_participant", "female_participant", "unknown_participant"]].groupby(GunVoilence_data["year"]).sum()
#dp_gen_plot=sns.kdeplot(genderwise_total['male_participant'], shade=True, color="r")
#dp_gen_plot=sns.kdeplot(genderwise_total['female_participant'], shade=True, color="b")
#dp_gen_plot=sns.kdeplot(genderwise_total['unknown_participant'], shade=True, color="g")
sns.distplot(genderwise_total['male_participant'], hist=False, rug=True);
sns.distplot(genderwise_total['female_participant'], hist=False, rug=True);
sns.distplot(genderwise_total['unknown_participant'], hist=False, rug=True,axlabel="participant");
#plt.label("Participant")

###########################Violin plot#####################################################
print(genderwise_total)
sns.violinplot( data=genderwise_total, split=True,         # If hue variable has two levels, draw half of a violin for each level.
               inner="box" );     # x-axis has categorical variable
#sns.violinplot( "split_frac", "gender", data=data );    # y-axis has categorical variable
del(genderwise_total)
print(yearly_impact)

sns.violinplot(data=yearly_impact[['n_killed','n_injured']],
               split=True,         # If hue variable has two levels, draw half of a violin for each level.
               inner="quartile"    #  Options: “box”, “quartile”, “point”, “stick”, None 
               )

 

#del(yearly_impact)
###############################Box plot###################
mth_injured_plot = sns.boxplot(GunVoilence_data["month"], "n_injured", data=yearly_impact[['n_killed','n_injured']])
mth_injured_plot.set_title("Person injured in injured/incidents per month/day") 
mth_killed_plot = sns.boxplot(GunVoilence_data["day"], "n_killed", data=yearly_impact[['n_killed','n_injured']])
mth_killed_plot.set_title("Person killed in injured/incidents per month/day") 
# Facet Grid Graphh ######################
#print(GunVoilence_data.weekday)
#print(GunVoilence_data[['n_killed']])
g2 = sns.FacetGrid(GunVoilence_data, hue="weekday", palette="Set1", size=5, hue_kws={"marker": ["^", "v"]})
g2.map(plt.scatter, "n_killed", "n_injured", s=100, linewidth=.5, edgecolor="white")
g2.add_legend();
