import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

from pandas import read_csv



%matplotlib inline

pd.set_option('display.max_rows', 20)
df = pd.read_csv("../input/ufo_sighting_data.csv")
df
df.info()
len(df)
df["country"]
df["country"].value_counts()
labelid = ["USA","CANADA","UK","AUSTRAALIA","SAKSAMAA"]



matplotlib.pyplot.pie(df["country"].value_counts(),labels=labelid,radius=3.5)
uus1 = pd.to_numeric(pd.Series(df["latitude"]), errors='coerce')

uus2 = pd.to_numeric(pd.Series(df["longitude"]), errors='coerce')

plt.scatter(uus1, uus2, 5, alpha=0.2)
uus = pd.to_numeric(pd.Series(df["length_of_encounter_seconds"]), errors='coerce', downcast='integer')

uus_sorted = uus.sort_values()



plt.figure(figsize=(40,20))

plt.locator_params(axis='x', numticks=10)

plt.xticks(fontsize=16, rotation=90)

plt.hist(df["length_of_encounter_seconds"], bins=100)
miinlat = uus1.mean()

miinlon = uus2.mean()

miinen = uus.mean()

print("Latitude'i mean: " + str(miinlat))

print("Longitude'i mean: " + str(miinlon))

print("Encounter'i mean: " + str(miinen))



#fd = pd.DataFrame({"Latitue mean" : miinlat,

#             "Longitude mean" : miinlon, "Encounter time mean": miinen})

fd = pd.DataFrame({"Latitue" : df["latitude"],

             "Longitude" : df["longitude"], "Encounter time": df["length_of_encounter_seconds"]})

fd
