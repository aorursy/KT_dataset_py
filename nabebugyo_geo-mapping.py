import pandas as pd
data = pd.read_csv("../input/chicago-grocery-stores-2013/grocery-stores-2013.csv")
data.head()
import matplotlib.pyplot as plt
%matplotlib inline
# Bounderies of Community Areas in Chicago
# Chicago Data Portal
# https://data.cityofchicago.org/
#
# Boundaries - Community Areas (current)
# https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Community-Areas-current-/cauq-8yn6
import geopandas as gpd
commareas = gpd.read_file("../input/chicago-community-areas-geojson/chicago-community-areas.geojson")

# Population
# City of Chicago
# https://www.cityofchicago.org/city/en/depts/dcd/supp_info/community_area_2000and2010censuspopulationcomparisons.html
import pandas as pd
population = pd.read_csv("../input/census-2010-and-2000-ca-populations/Census_2010_and_2000_CA_Populations.csv")

population["Upper"] = population.apply(lambda row: row["Community Area"].upper(), axis=1)
commareas_ppl = pd.merge(commareas, population, how="left", left_on="community", right_on="Upper")
commareas_ppl = commareas_ppl[["Community Area", "community", "geometry", "2010"]]

import matplotlib.pyplot as plt
%matplotlib inline

commareas_ppl.fillna(0).plot(column="2010", cmap="gray_r", alpha=0.4, edgecolor="black", figsize=(12, 12))

#plt.scatter(x=data["LONGITUDE"], y=data["LATITUDE"], c=data["SQUARE FEET"], cmap="jet", marker="x")
plt.scatter(x=data["LONGITUDE"], y=data["LATITUDE"], c=data["SQUARE FEET"], cmap="viridis", s=data["SQUARE FEET"]/200, alpha=0.5)

for x, y, t in zip(commareas.centroid.x, commareas.centroid.y, commareas.community):
    plt.text(x=x, y=y, s=t, size=7, color="gray",
                    horizontalalignment='center', verticalalignment='center')

plt.colorbar(fraction=0.01, pad=-0.12).ax.text(x=0, y=1.05, s="SQUARE FEET")
#fig, axes = plt.subplots(1, 2, figsize=(20, 5))
fig = plt.figure(figsize=(20,5))
ax1 = plt.subplot2grid((1,5), (0,0), colspan=4)
ax2 = plt.subplot2grid((1,5), (0,4), colspan=1)

nstore = len(data["COMMUNITY AREA NAME"].unique())
q1 = round(nstore*1/5)
q2 = round(nstore*4/5)-q1
q3 = nstore-q1-q2

#plt.xlabel("COMMUNITY AREA NAME")
#plt.ylabel("Freqency")


tmp = data["COMMUNITY AREA NAME"].value_counts()

#box = tmp.plot(kind="box", ax=ax2)
box = ax2.boxplot(tmp)
ax2.set_xlabel("COMMUNITY AREA NAME")

y0 = box["whiskers"][0].get_ydata()[0]
y1 = box["whiskers"][1].get_ydata()[0]

tmp = pd.DataFrame(tmp)
tmp["color"] = "cornflowerblue"
tmp["color"] = tmp.apply(lambda row: "blue" if row["COMMUNITY AREA NAME"] > y1 else row["color"], axis=1)
tmp["color"] = tmp.apply(lambda row: "lightblue" if row["COMMUNITY AREA NAME"] < y0 else row["color"], axis=1)
colors = tmp["color"].tolist()
ax2.set_xticklabels("")

tmp["COMMUNITY AREA NAME"].plot(kind="bar", color=colors, ax=ax1)
ax1.set_ylabel("Frequency")

# Drawing bounderies of Community Areas in Chicago
# Chicago Data Portal
# https://data.cityofchicago.org/
#
# Boundaries - Community Areas (current)
# https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Community-Areas-current-/cauq-8yn6
import geopandas as gpd
commareas = gpd.read_file("../input/chicago-community-areas-geojson/chicago-community-areas.geojson")

import pandas as pd
population = pd.read_csv("../input/census-2010-and-2000-ca-populations/Census_2010_and_2000_CA_Populations.csv")

population["Upper"] = population.apply(lambda row: row["Community Area"].upper(), axis=1)
commareas_ppl = pd.merge(commareas, population, how="left", left_on="community", right_on="Upper")
commareas_ppl = commareas_ppl[["Community Area", "community", "geometry", "2010"]]

import matplotlib.pyplot as plt
%matplotlib inline

commareas_ppl.fillna(0).plot(column="2010", cmap="gray_r", alpha=0.4, edgecolor="black", figsize=(12, 12))

for x, y, t in zip(commareas.centroid.x, commareas.centroid.y, commareas.community):
    if t in tmp.index.tolist():
        c = tmp["color"][t]
    else:
        c = "gray"
    plt.text(x=x, y=y, s=t, size=7, color=c,
                    horizontalalignment='center', verticalalignment='center')

data["SQUARE FEET"].plot(kind="hist", color="cornflowerblue")
plt.xlabel("SQUARE FEET")
commareas_ppl.fillna(0).plot(column="2010", cmap="gray_r", alpha=0.4, edgecolor="black", figsize=(12, 12))

plt.scatter(x=data["LONGITUDE"], y=data["LATITUDE"], c="powderblue", marker="x")

tmp = data[data["SQUARE FEET"] > 20000].copy()
plt.scatter(x=tmp["LONGITUDE"], y=tmp["LATITUDE"], c="red", marker="x")

for x, y, t in zip(commareas.centroid.x, commareas.centroid.y, commareas.community):
    plt.text(x=x, y=y, s=t, size=7, color="gray",
                    horizontalalignment='center', verticalalignment='center')

plt.legend(["Others", "Large stores (SQUARE FEET > 40000)"])
data["STORE NAME"].value_counts()[data["STORE NAME"].value_counts()>1].plot(kind="bar", color="cornflowerblue")
plt.xlabel("STORE NAME (Frequency > 1)")
plt.ylabel("Frequency")
commareas_ppl.fillna(0).plot(column="2010", cmap="gray_r", alpha=0.4, edgecolor="black", figsize=(12, 12))

plt.scatter(x=data["LONGITUDE"], y=data["LATITUDE"], c="powderblue", marker="x")

over4 = ["WHOLE FOODS", "CERMAK", "TONY'S FINER", "TREASURE ISLAND"]
colorlist = ["red", "blue", "yellow", "green"]
for store, c in zip(over4, colorlist):
    tmp = data[data["STORE NAME"].str.contains(store)].copy()
    plt.scatter(x=tmp["LONGITUDE"], y=tmp["LATITUDE"], c=c, marker="x")

for x, y, t in zip(commareas.centroid.x, commareas.centroid.y, commareas.community):
    plt.text(x=x, y=y, s=t, size=7, color="gray",
                    horizontalalignment='center', verticalalignment='center')
plt.legend(("Others", "WHOLE FOODS", "CERMAK PRODUCE", "TONY'S FINER", "TREASURE ISLAND"))
