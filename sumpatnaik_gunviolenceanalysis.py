import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# To show graph in same window.
%matplotlib inline
#gunData = pd.read_csv("C:\\Users\\sumpatna\\Desktop\\AI-Bigdata\\kaggleExercise\\csv\\gun-violence-data_01-2013_03-2018.csv")
gunData=pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")
gunData.head(5)
# Check datatypes
gunData.dtypes
(239677, 29)
# Drop the columns from the dataset which maynot be useful
gunData.drop([
    "incident_url",
    "sources",
    "source_url",
    "incident_url_fields_missing",
    "location_description",
    ], axis=1, inplace=True)

# Adding  new columns like Day, Month, Year, Week, Weekday for better analysis

gunData['date'] = pd.to_datetime(gunData['date']) 
gunData["day"] = gunData["date"].dt.day
gunData["month"] = gunData["date"].dt.month
gunData["year"] = gunData["date"].dt.year
gunData["week"] = gunData["date"].dt.week
gunData["weekday"] = gunData["date"].dt.weekday
gunData["quarter"] = gunData["date"].dt.quarter
# splitting the particpitant_gender to their respectives gender groupz
gunData["participant_gender"] = gunData["participant_gender"].fillna("0::Unknown")
    
def gen(n) :                    
    genrow = []               
    genrow = str(n).split("||")    
    for i in genrow :              
        g_row = str(i).split("::")  
        if len(g_row) > 1 :         
            genrow.append(g_row[1])    

    return genrow

pGender = gunData.participant_gender.apply(gen)
gunData["totalParticipant"] = pGender.apply(lambda x: len(x))
gunData["maleParticipant"] = pGender.apply(lambda i: i.count("Male"))
gunData["femaleParticipant"] = pGender.apply(lambda i: i.count("Female"))
gunData["unknownParticipant"] = pGender.apply(lambda i: i.count("Unknown"))
gunData.head()

mpl.style.use("seaborn")
plt.style.use("seaborn")
###### Now Start data ploting#######
# Data joint plot on number of person killed on yearly basis
sns.jointplot("year","n_injured",gunData,kind='scatter',
              s=200, color='m', edgecolor="black", linewidth=3)
# Data joint plot on number of person killed per state
sns.jointplot("month","n_injured",gunData,kind='scatter',
              s=200, color='m', edgecolor="black", linewidth=3)
# Plot Histogram for Top 10 state with maximum killed of Gun Violence

stateTotalkilled= gunData[["n_killed"]].groupby(gunData["state"]).sum()
topStateKilled = stateTotalkilled.sort_values(by='n_killed', ascending=False).head(25)
print(topStateKilled)
topStateKilled.plot.barh()
del(topStateKilled)


stateTotalInjured= gunData[["n_injured"]].groupby(gunData["state"]).sum()
topStateInjured = stateTotalInjured.sort_values(by='n_injured', ascending=False).head(25)
print(topStateInjured)
topStateInjured.plot.barh()
del(topStateInjured)


# Plot Histogram for Top 10 city with maximum killed of Gun Violence

cityOrCounty= gunData[["n_killed"]].groupby(gunData["city_or_county"]).sum()
topCityOrCounty = cityOrCounty.sort_values(by='n_killed', ascending=False).head(10)
print(topCityOrCounty)
topCityOrCounty.plot.barh()
del(topCityOrCounty)
#draw histogram  plot for yearly incident,injured and killed 

incidentdata= gunData[["n_killed","n_injured"]].groupby(gunData["year"]).sum()
print(incidentdata)
incidentdata.plot.barh()



#######Kernal Density plot #########

#To draw plot for yearly incident,injured and killed 

densityPlot=sns.kdeplot(incidentdata['n_killed'],shade=True,color="blue")
densityPlot=sns.kdeplot(incidentdata['n_injured'],shade=True,color="green")
print(incidentdata['n_killed'])
sns.distplot(incidentdata['n_killed'], hist=False, rug=True);

# Density graph for gender participant

incidentdata= gunData[["maleParticipant","femaleParticipant"]].groupby(gunData["year"]).sum()
densityPlot=sns.kdeplot(incidentdata['maleParticipant'],shade=True,color="yellow")
densityPlot=sns.kdeplot(incidentdata['femaleParticipant'],shade=True,color="red")

## Violin Plot for  Person Killed per year
yearKilled = sns.violinplot("year", "n_killed", data=gunData,
                                 split=True, inner="quartile")
yearKilled.set_title("Person killed per Year")

## Violin Plot for  Person injured per year
yearKilled = sns.violinplot("year", "n_injured", data=gunData,
                                 split=True, inner="quartile")
yearKilled.set_title("Person injured per Year")
###################################  Box plots ##################################
## Box Plot for  Person Killed per year

personKilled = sns.boxplot("year", "n_killed", data= gunData)
personKilled.set_title("Person killed per year")
## Box Plot for  Person injured per year
personInjured = sns.boxplot("year", "n_injured", data= gunData)
personInjured.set_title("Person injured per year")
#Facet Grid
# number of person killed per year

sns.set(style="ticks")
df = pd.DataFrame(gunData)
g = sns.FacetGrid(df, col = "year")
g.map(plt.hist, "n_killed")
plt.show()

df = pd.DataFrame(gunData)
g = sns.FacetGrid(df, col = "year")
g.map(plt.scatter,"year", "n_killed")
