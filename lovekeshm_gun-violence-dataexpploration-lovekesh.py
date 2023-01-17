import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
#Read data
data = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")
#data Analysis
data.head()
data.shape
data.info()
data.describe()
data.columns
#find any null values
data.isnull().values.any()
data.isnull().sum()
#convert data column
data["date"] = pd.to_datetime(data["date"])
data["day"] = data["date"].dt.day
data["month"] = data["date"].dt.month
data["year"] = data["date"].dt.year
data["weekday"] = data["date"].dt.weekday
data['address']
#Removing data that we are not going to us
data.drop(["incident_url", "source_url", "sources", "incident_url_fields_missing", "latitude", "location_description", "longitude", "participant_relationship"], axis=1, inplace=True)
#statewise crimerate
data_cplot=sns.countplot("state", data = data, order=data["state"].value_counts().index,orient="v",palette="Set3")
data_cplot.set_xticklabels(data_cplot.get_xticklabels(),rotation=90)
#citywise crime rate
CityData=data['city_or_county'].value_counts().head(25)
plt.pie(CityData, labels=CityData.index, shadow=True, startangle=120)
#How many people were killed each year
data_year_killed=data.groupby(data["year"]).apply(lambda x: pd.Series(dict(killed_year=x.n_killed.sum())))
data_year_killed
#Plotting yearly killed
data_year_killed_plot= sns.pointplot(x=data_year_killed.index, y=data_year_killed.killed_year, data=data_year_killed,label="Victims killed per year")
#How many people were killed each year
data_year_injured=data.groupby(data["year"]).apply(lambda x: pd.Series(dict(injured_year=x.n_injured.sum())))
data_year_injured
data_year_injured_plot= sns.barplot(x=data_year_injured.index, y=data_year_injured.injured_year, data=data_year_injured,label="Victims injured per year")
data_year_injured_plot.set_xticklabels(data_year_injured_plot.get_xticklabels(),rotation=45)
data_year_injured_plot.set_title("Victims injured per year")
#Plotting monthly killed and injured

data_month_killed=data.groupby(data["month"]).apply(lambda x: pd.Series(dict(killed_month=x.n_killed.sum())))
data_month_killed_plot= sns.barplot(x=data_month_killed.index, y=data_month_killed.killed_month)
data_month_killed_plot.set_xticklabels(data_month_killed_plot.get_xticklabels(),rotation=90)
data_month_killed_plot.set_title("Victims killed per month")


data_month_injured=data.groupby(data["month"]).apply(lambda x: pd.Series(dict(injured_month=x.n_injured.sum())))
data_month_injured_plot= sns.barplot(x=data_month_injured.index, y=data_month_injured.injured_month)
data_month_injured_plot.set_xticklabels(data_month_injured_plot.get_xticklabels(),rotation=90)
data_month_injured_plot.set_title("Victims injured per month")
#Jointplot to show killed vs injured for all incidents
sns.jointplot("n_injured","n_killed",data,kind='scatter', s=400, color='r', edgecolor="grey", linewidth=3)
#We can also show victims killed year in a histogram
data_year_killed=data.groupby(data["year"]).apply(lambda x: pd.Series(dict(killed_year=x.n_killed.sum())))
data_year_killed.plot.barh()
#Lets check gun_stolen and n_guns_involved values
data['gun_stolen'].unique()

data['n_guns_involved'].unique()
data['gun_type'].unique()
#fill na values for these columns
data["n_guns_involved"] = data["n_guns_involved"].fillna(0)
data["gun_stolen"] = data["gun_stolen"].fillna("0::Unknown")
def stolgun(row) :
    unknown = 0
    stolen = 0
    notstolen = 0
    row_values = []
    
    row = str(row).split("||")
    for x in row :
            row_value = str(x).split("::")
            if len(row_value) > 1 :
                row_values.append(row_value[1])
                if "Stolen" in row_value :
                    stolen += 1
                elif "Not-stolen" in row_value :
                    notstolen += 1
                else :
                    unknown += 1
                    
    return row_values
gunstolen = data.gun_stolen.apply(stolgun)
data["stolen_gun"] = gunstolen.apply(lambda x: x.count("Stolen"))
data["notstolen_gun"] = gunstolen.apply(lambda x: x.count("Not-stolen"))
data.head(5)
#dentisty plot
Gun_stolen_notstolen = data[["stolen_gun", "notstolen_gun"]].groupby(data["year"]).sum()
stolen_den_plot=sns.kdeplot(Gun_stolen_notstolen['stolen_gun'], shade=True, color="r")
notstolen_plot=sns.kdeplot(Gun_stolen_notstolen['notstolen_gun'], shade=True, color="b")
#Violin plots
impacted_people = data[["n_killed","n_injured"]].groupby(data["year"]).sum()
impacted_people=sns.violinplot(data=impacted_people,split=True,inner="quartile")
