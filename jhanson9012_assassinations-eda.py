import base64

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

import numpy as np 

import os

import pandas as pd 

import seaborn as sns



print(os.listdir("../input"))
dataset = pd.read_csv("../input/globalterrorismdb_0718dist.csv", encoding='ISO-8859-1')



# Take only assassinations

asn = dataset[dataset["attacktype1"] == 1]



# List of variables to keep

keep = ["eventid", "iyear", "imonth", "iday", "country_txt", "region_txt", "provstate", "latitude", "longitude", "success", "suicide", "attacktype1_txt",  "targtype1_txt", "targsubtype1_txt", "gname", "weaptype1_txt",

       "weapsubtype1_txt"]

asn = asn[keep] 



# Rename columns and rows as appropriate

asn.rename(columns={'iyear':'year','imonth':'month','iday':'day','country_txt':'country','region_txt':'region', "provestate":"province_state",

                       "attacktype1_txt":"attack_type", "targtype1_txt":"target_type", "targsubtype1_txt":"target_subtype", "gname":"group_name", "weaptype1_txt":"weapon_type",

                       "weapsubtype1_txt":"weapon_subtype"}, inplace = True)



asn["weapon_type"].replace({

        "Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)":"Vehice"},

inplace = True)



asn["weapon_subtype"].replace({

        "Automatic or Semi-Automatic Rifle":"Rifle (auto or semi)", 

        "Unknown Gun Type":"Unknown", 

        "Rifle/Shotgun (non-automatic)":"Rifle/Shotgun (non-auto)",

        "Knife or Other Sharp Object":"Knife"},

inplace = True)
asn.head(5)
plt.figure(figsize=(15,6))

sns.countplot("year", data = asn, hue = "success")

plt.xticks(rotation = 90)

plt.ylabel("Number of Assassinations")

plt.xlabel("Year")

plt.title("Assassinations by Year")

plt.show()
plt.figure(figsize=(15,6))

sns.countplot(asn.region, order = asn['region'].value_counts().index)

plt.xlabel("Region")

plt.xticks(rotation = 90)

plt.ylabel("Number of Assassinations")

plt.title("Assassinations by Region")

plt.show()
by_region = pd.crosstab(asn.year,asn.region)

by_region.plot(color=sns.color_palette('Set1',12))

figure = plt.gcf()

figure.set_size_inches(15,6)

plt.ylabel("Number of Assassinations")

plt.xlabel("Year")

plt.title("Assassinations by Region, 1970 to 2017")

plt.legend(bbox_to_anchor=(1.02,0.5), loc="center left")

plt.show()

# Take out unknown weapon types, other, and sabotage equipment (only one data point; it was not successful)

drop = ["Unknown", "Other", "Sabotage Equipment"]

weapons = asn[~asn["weapon_type"].isin(drop)]



plt.figure(figsize = (15,6))

sns.countplot(weapons.weapon_type, order = weapons['weapon_type'].value_counts().index)

plt.xticks(rotation = 90)

plt.xlabel("Weapon Type")

plt.ylabel("Number of Assassinations")

plt.title("Assassinations by Weapon Type")

plt.show()

drop = ["Unknown", "Other", "Sabotage Equipment"]

weapons = asn[~asn["weapon_type"].isin(drop)]



by_weapon = pd.crosstab(weapons.weapon_type, weapons.success)

by_weapon["rate"] = by_weapon[1] / (by_weapon[0] + by_weapon[1])



plt.figure(figsize = (15,6))

sns.barplot(by_weapon.index, by_weapon.rate)

plt.xlabel("Weapon Type")

plt.xticks(rotation = 90)

plt.ylabel("Success Rate")

plt.title("Success Rates of Weapon Types")

plt.show()
fa = asn[asn["weapon_type"] == "Firearms"]

fa = fa[fa["weapon_subtype"] != "Unknown"]



plt.figure(figsize = (15,5))

p = sns.countplot(fa.weapon_subtype, hue = fa.success, 

                  order = fa.weapon_subtype.value_counts().index)

p.set_xticklabels(fa["weapon_subtype"].unique())

plt.xlabel("Firearm Type")

plt.ylabel("Number of Assassinations")

plt.title("Most common Firearm Types")

plt.show()
plt.figure(figsize = (15,5))

p = sns.countplot(asn.target_type, hue = asn.success, 

              order = asn.target_type.value_counts().index)

p.set_xticklabels(asn["target_type"].unique(), rotation = 90)

plt.xlabel("Target Types")

plt.ylabel("Number ofAssassinations")

plt.title("Most Common Target Types")

plt.show()
priv_prop = asn[asn["target_type"] == "Private Citizens & Property"]



plt.figure(figsize = (15,5))

p = sns.countplot(priv_prop.target_subtype, hue = priv_prop.success, 

              order = priv_prop.target_subtype.value_counts().index)

plt.xlabel("Private Citizen and Property Subtype")

plt.ylabel("Number ofAssassinations")

plt.title("Most Common Target Subtypes for Private Citizens and Property")



p.set_xticklabels(asn["target_subtype"].unique(), rotation = 90)

plt.show()
lat = asn["latitude"].values

lon = asn["longitude"].values





fig = plt.figure(num=None, figsize=(16,9))

m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,resolution='c')

m.drawcoastlines()

m.drawcountries()

lons, lats = m(lon, lat)

m.scatter(lons, lats, marker = "o", color = "r")

plt.show()
