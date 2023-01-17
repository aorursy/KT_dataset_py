#Gun Violence Data Exploration, Analysis and Plotting

# At first import following libraries

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# To show graph in same window.
%matplotlib inline
mpl.style.use("seaborn")
plt.style.use("seaborn")
# Read the Gun violence dataset to analyse the data.

gunv_data = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")
gunv_data.head()


gunv_data.columns.values
gunv_data.dtypes
gunv_data.describe()
gunv_data.info()
gunv_data.shape
# Drop following extra columns from the dataset
gunv_data.drop([
    "incident_url",
    "sources",
    "source_url",
    "incident_url_fields_missing",
    "location_description",
    "participant_relationship",
    ], axis=1, inplace=True)
# Convert date column datatype into datetime
gunv_data['date'] = pd.to_datetime(gunv_data['date']) 

# Now need some addition columns from date for advance analysis.
# Adding some new columns like Day, Month, Year, Week, Weekday, etc

gunv_data["day"] = gunv_data["date"].dt.day
gunv_data["month"] = gunv_data["date"].dt.month
gunv_data["year"] = gunv_data["date"].dt.year
gunv_data["week"] = gunv_data["date"].dt.week
gunv_data["weekday"] = gunv_data["date"].dt.weekday
gunv_data["quarter"] = gunv_data["date"].dt.quarter

# Checking for null value.
gunv_data["n_guns_involved"] = gunv_data["n_guns_involved"].fillna(value =0)
gunv_data["gun_stolen"] = gunv_data["gun_stolen"].fillna(value = "0::Unknown")
## Creating multiple columns from Participant's Gender column
gunv_data["participant_gender"] = gunv_data["participant_gender"].fillna("0::Unknown")
    
def gen(n) :                    
    genrow = []               
    genrow = str(n).split("||")    
    for i in genrow :              
        g_row = str(i).split("::")  
        if len(g_row) > 1 :         
            genrow.append(g_row[1])    

    return genrow

p_gender = gunv_data.participant_gender.apply(gen)
gunv_data["total_participant"] = p_gender.apply(lambda x: len(x))
gunv_data["male_participant"] = p_gender.apply(lambda i: i.count("Male"))
gunv_data["female_participant"] = p_gender.apply(lambda i: i.count("Female"))
gunv_data["unknown_participant"] = p_gender.apply(lambda i: i.count("Unknown"))
# check the new dataset
gunv_data.head()

###### Now Start data ploting#######
# Data joint plot on number of person killed vs no of person injured
sns.jointplot("n_injured","n_killed",gunv_data,kind='scatter',
              s=200, color='m', edgecolor="yellow", linewidth=3)
# Data joint plot on number of person killed from which state
sns.jointplot("month","n_killed",gunv_data,kind='scatter',
              s=200, color='m', edgecolor="yellow", linewidth=2)
# Data joint plot on number of gun involved and no of guns stolen.
sns.jointplot("year","n_killed",gunv_data,kind='scatter',s = 200,
             color='m', edgecolor="yellow", linewidth=2)
###############################  Histograms  #########################
# Plot Histogram for Top 10 state with maximum incidents of Gun Violence

state_wise_total= gunv_data[["incident_id"]].groupby(gunv_data["state"]).count()
top_state = state_wise_total.sort_values(by='incident_id', ascending=False).head(10)
print(top_state)
top_state.plot.barh()
del(top_state)
###############################  Histograms  #########################
# Histogram for Top 10 Cities with maximum incidents of Gun Violence

city_wise_total= gunv_data[["incident_id"]].groupby(gunv_data["city_or_county"]).count()
top_city = city_wise_total.sort_values(by='incident_id', ascending=False).head(10)
print(top_city)
top_city.plot.barh()
del(top_city)
###############################  Histograms  #########################
# Plot Histogram for year with maximum incidents of Gun Violence

year_wise_total= gunv_data[["incident_id"]].groupby(gunv_data["year"]).count()
top_year = year_wise_total.sort_values(by='incident_id', ascending=False)
print(top_year)
top_year.plot.barh()
del(top_year)
#######Kernal Density plot #########

#To draw plot for yearly incident,injured and killed 

year_wise = gunv_data[["n_killed", "n_injured"]].groupby(gunv_data["year"]).sum()
density_plot=sns.kdeplot(year_wise['n_killed'],shade=True,color="red")
density_plot=sns.kdeplot(year_wise['n_injured'],shade=True,color="blue")
print(year_wise['n_killed'])
sns.distplot(year_wise['n_killed'], hist=False, rug=True);
# Density plot for gendrwise participant
genderwise_total = gunv_data[["total_participant", "male_participant", "female_participant", "unknown_participant"]].groupby(gunv_data["year"]).sum()
dp_gen_plot=sns.kdeplot(genderwise_total['male_participant'], shade=True, color="r")
dp_gen_plot=sns.kdeplot(genderwise_total['female_participant'], shade=True, color="b")
dp_gen_plot=sns.kdeplot(genderwise_total['unknown_participant'], shade=True, color="g")
del(genderwise_total)
################################## Violin plots #################################
## Violin Plot for Yearwise Person Injured
year_injured = sns.violinplot("year", "n_injured", data=gunv_data,
                                 split=True, inner="quartile")
year_injured.set_title("Person injured per Year")

################################## Violin plots #################################
## Violin Plot for Year-wise Person Killed
year_killed = sns.violinplot("year", "n_killed", data=gunv_data,
                                 split=True, inner="quartile")
year_killed.set_title("Person killed per Year")

#### Person impected injured,killed yearly
yearly_impact = gunv_data[["n_killed","n_injured"]].groupby(gunv_data["year"]).sum()
print(yearly_impact)
yearly_impact=sns.violinplot(data=yearly_impact,split=True,inner="quartile")
###################################  Box plots ##################################
#Monthwise Person injured
monthwise_prsn_injured = sns.boxplot("month", "n_injured", data= gunv_data)
monthwise_prsn_injured.set_title("Month wise total person injured in incident")
###################################  Box plots ##################################
#Monthwise Person Killed
monthwise_prsn_kild = sns.boxplot("month", "n_killed", data= gunv_data)
monthwise_prsn_kild.set_title("Month wise total person killed in incident")
###################################  Box plots ##################################
#Year wise Person injured
yearwise_prsn_injured = sns.boxplot("year", "n_injured", data= gunv_data)
yearwise_prsn_injured.set_title("Year wise total person injured in incident")
###################################  Box plots ##################################
#Year wise Person injured
yearwise_prsn_injured = sns.boxplot("year", "n_injured", data= gunv_data)
yearwise_prsn_injured.set_title("Year wise total person injured in incident")