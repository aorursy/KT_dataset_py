import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import geopandas as gpd
import pandas as pd
def plot_record(shape, df, date):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    shape["Case"] = 0
    ax.set_facecolor("gray")
    
    coords = {
        "Istanbul": (29, 41), 
        "Western Marmara": (27.5, 39.7),
        "Aegean": (28.5, 38),
        "Eastern Marmara": (30, 40),
        "Western Anatolia": (32.5, 38),
        "Mediterranean": (32.5, 36),
        "Central Anatolia": (35.5, 39),
        "Western Blacksea": (34.5, 40.8),
        "Eastern Blacksea": (39.2, 40.5),
        "Northeastern Anatolia": (41.5, 39.8),
        "Mideastern Anatolia": (40.5, 38.8),
        "Southeastern Anatolia": (40, 37.4)
    }
    
    for column in df.columns[1:-1]:
        case_count = df.loc[df["Date"] == date, column].values[0]
        shape.loc[shape["Region"] == column, "Case"] = case_count
        plt.annotate(s=case_count, xy=coords[column], horizontalalignment='center', color="white", fontsize=30)
        
    shape.plot(column="Case", ax=ax, legend=True, cmap='copper', legend_kwds={'label': "Case count", 'orientation': "horizontal", 'shrink': 0.3})
    shape.drop(columns=["Case"])
turkey_map = gpd.read_file("../input/covid19-in-turkey-by-regions/shape/turkey.shp")
table = pd.read_csv("../input/covid19-in-turkey-by-regions/turkey_covid19.csv")
date = "2020-08-31"

plot_record(turkey_map, table, date)
fig, ax = plt.subplots(1, 1, figsize=(20, 12))

table["Date"] = pd.to_datetime(table["Date"], yearfirst=True)

ax.set_title("Covid19 Cases by Regions")
ax.set_xlabel("Date")
ax.set_ylabel("Cases")

locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)

ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

for column in table.columns[1:-1]:
    ax.plot(table["Date"], table[column])
    
ax.legend(table.columns[1:-1])

ax.grid()

plt.show()
