import numpy as np
import pandas as pd
import seaborn as sns
from dask import dataframe as dd
from datetime import datetime as dt
from matplotlib import pyplot as plt
DAY_NAMES = ["SUNDAY", "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY"]
MONTH_NAMES = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]
RDM_SEED = 345
INTERVENTIONS = dd.read_csv("../*/SIM Interventions (*).csv").compute().reset_index(drop = True)
INTERVENTIONS_COUNT = INTERVENTIONS.shape[0]

print("{:,d} SIM Interventions".format(INTERVENTIONS_COUNT))
INTERVENTIONS.sample(20, random_state = RDM_SEED)
INTERVENTIONS = INTERVENTIONS.drop("TYPE", axis = 1)

INTERVENTIONS["DATETIME"] = pd.to_datetime(INTERVENTIONS["DATETIME"], format = "%d/%m/%Y %H:%M")
INTERVENTIONS["YEAR"] = INTERVENTIONS["DATETIME"].apply(lambda dt : dt.strftime("%Y"))
INTERVENTIONS["MONTH"] = INTERVENTIONS["DATETIME"].apply(lambda dt : dt.strftime("%B").upper())
INTERVENTIONS["DAY"] = INTERVENTIONS["DATETIME"].apply(lambda dt : dt.strftime("%A").upper())
INTERVENTIONS["DATE"] = INTERVENTIONS["DATETIME"].apply(lambda dt : dt.strftime("%x"))
INTERVENTIONS["TIME"] = INTERVENTIONS["DATETIME"].apply(lambda dt : dt.strftime("%X"))
INTERVENTIONS = INTERVENTIONS.drop("DATETIME", axis = 1)

INTERVENTIONS.sample(20, random_state = RDM_SEED)
INTERVENTIONS.dtypes
INTERVENTIONS["LAT"] = INTERVENTIONS["LAT"].str.replace(",", ".").astype(float)
INTERVENTIONS["LON"] = INTERVENTIONS["LON"].str.replace(",", ".").astype(float)

INTERVENTIONS.dtypes
INTERVENTIONS.isna().any()
INTERVENTIONS[INTERVENTIONS["LAT"].isna() | INTERVENTIONS["LON"].isna()]
INTERVENTIONS = INTERVENTIONS.dropna()
def display_interventions_by_type_plot(interventions):
    sns.set_style("whitegrid")
    plt.figure(figsize = (20, 6))
    ax = sns.countplot(data = interventions, x = "DESC_TYPE", order= interventions["DESC_TYPE"].value_counts().index)

    for p in ax.patches:
        x = p.get_bbox().get_points()[:,0]
        y = p.get_bbox().get_points()[1,1]
        ax.annotate(
            '{:.2f}%'.format(y / len(interventions) * 100), 
            (x.mean(), y), 
            ha = 'center', 
            va = 'bottom'
       )

    sns.despine(trim = True)

    plt.title("Interventions By Type")
    plt.xlabel("Intervention Type")
    plt.ylabel("Number of Interventions")
    plt.show()
    
display_interventions_by_type_plot(INTERVENTIONS)
INTERVENTIONS = INTERVENTIONS[INTERVENTIONS["DESC_TYPE"] != "Premier répondant"]

INTERVENTION_TYPES = ['Sans incendie', 'Alarmes-incendies', 'Autres incendies', 'Fausses alertes/annulations', 'Incendies de bâtiments']

display_interventions_by_type_plot(INTERVENTIONS)
def display_interventions_by_year_plot(interventions):
    sns.set_style("whitegrid")
    plt.figure(figsize = (20, 6))
    ax = sns.countplot(data = interventions, x = "YEAR")

    for p in ax.patches:
        x = p.get_bbox().get_points()[:,0]
        y = p.get_bbox().get_points()[1,1]
        ax.annotate(
            '{:.2f}%'.format(y / len(interventions) * 100), 
            (x.mean(), y), 
            ha = 'center', 
            va = 'bottom'
       )

    sns.despine(trim = True)

    plt.title("Interventions By Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Interventions")
    plt.xticks(rotation = 90)
    plt.show()
    
display_interventions_by_year_plot(INTERVENTIONS)
INTERVENTIONS = INTERVENTIONS[INTERVENTIONS["YEAR"] != "2018"]
display_interventions_by_year_plot(INTERVENTIONS)
def display_interventions_by_day_and_month_plot(interventions):
    sns.set_style("whitegrid")
    plt.figure(figsize = (15, 12))
    cross = pd.crosstab(interventions["MONTH"], interventions["DAY"])
    cross = cross.reindex(index = MONTH_NAMES, columns = DAY_NAMES)

    sns.heatmap(cross, annot = True, fmt = ",d", cmap = "coolwarm")
    
    plt.show()
    
display_interventions_by_day_and_month_plot(INTERVENTIONS)
def display_interventions_by_borough_plot(interventions):
    interventions = interventions[~interventions["BOROUGH"].str.contains("Indéterminé")]
    sns.set_style("whitegrid")
    plt.figure(figsize = (20, 6))
    ax = sns.countplot(data = interventions, x = "BOROUGH", order = interventions["BOROUGH"].value_counts().index)

    for p in ax.patches:
        x = p.get_bbox().get_points()[:,0]
        y = p.get_bbox().get_points()[1,1]
        ax.annotate(
            '{:.1f}%'.format(y / len(interventions) * 100), 
            (x.mean(), y), 
            ha = 'center', 
            va = 'bottom'
       )

    sns.despine(trim = True)

    plt.title("Interventions By Borough")
    plt.xlabel("Borough")
    plt.ylabel("Number of Interventions")
    plt.xticks(rotation = 90)
    plt.show()
    
display_interventions_by_borough_plot(INTERVENTIONS)
import folium
from folium import plugins
from folium.plugins import HeatMap, HeatMapWithTime

montreal_map = folium.Map(tiles = "Stamen Terrain", location = [45.5017, -73.5673], zoom_start = 12) 

years = INTERVENTIONS["YEAR"].unique()
# List comprehension to make out list of lists
locations = [[[row['LAT'], row['LON']] for index, row in INTERVENTIONS[INTERVENTIONS["YEAR"] == year][["LAT", "LON"]].sample(4000).iterrows()] for year in years]

# Plot it on the map
montreal_heatmap = HeatMapWithTime(locations, radius = 12)
montreal_heatmap.add_to(montreal_map)

# Display the map
montreal_map
import folium
from folium import plugins
from folium.plugins import HeatMap, HeatMapWithTime

montreal_map = folium.Map(tiles = "Stamen Terrain", location = [45.5017, -73.5673], zoom_start = 12) 

# List comprehension to make out list of lists
locations = [[row['LAT'], row['LON']] for index, row in INTERVENTIONS[INTERVENTIONS["DESC_TYPE"] == "Incendies de bâtiments"][["LAT", "LON"]].iterrows()]

# Plot it on the map
HeatMap(locations, radius = 12).add_to(montreal_map)

# Display the map
montreal_map
INTERVENTIONS_BY_DATE = INTERVENTIONS.groupby("DATE").size().to_frame("COUNT").reset_index()
INTERVENTIONS_BY_DATE["DATE_ORD"] = pd.to_datetime(INTERVENTIONS_BY_DATE["DATE"]).apply(lambda date : date.toordinal())

INTERVENTIONS_FIRST_DATE = INTERVENTIONS_BY_DATE["DATE_ORD"].min()
INTERVENTIONS_LAST_DATE = INTERVENTIONS_BY_DATE["DATE_ORD"].max()

plt.figure(figsize = (20, 8))
ax = sns.regplot(
    data = INTERVENTIONS_BY_DATE, 
    x = "DATE_ORD", y = "COUNT", 
    logx = True, 
    scatter_kws = {'alpha' : 0.3}, 
    line_kws = {"color" : "red"}
)

# Tighten up the axes for prettiness
ax.set_xlim(INTERVENTIONS_FIRST_DATE - 1, INTERVENTIONS_LAST_DATE + 1)
ax.set_ylim(0, INTERVENTIONS_BY_DATE['COUNT'].max() + 1)

ax.set_xlabel("DATES")
new_labels = [dt.strftime(dt.fromordinal(int(item)), "%B, %Y") for item in ax.get_xticks()]
ax.set_xticklabels(new_labels)

plt.show()
