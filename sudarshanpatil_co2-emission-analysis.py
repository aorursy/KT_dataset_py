import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings as ws

ws.filterwarnings("ignore")
df = pd.read_csv("/kaggle/input/co2-emission-by-vehicles/CO2 Emissions_Canada.csv")
df.head()
df.columns.to_list()
# clean Column name 

def clean_name(name):

    return name.strip().lower().replace("/", "_per_").replace(" ", "_")
df.rename(columns = clean_name, inplace = True)
df.head()
df.isna().sum()
sns.set()

plt.title("vehicle_class in the canada")

sns.barplot(y="index", x = "vehicle_class", data = df.vehicle_class.value_counts().reset_index(), palette="inferno")

plt.show()
# Calculate No. of cylinders in the car

temp = df.cylinders.value_counts().reset_index().rename(columns = {"index": "no_of_cylinders", "cylinders": "count"})

sns.set()

sns.catplot(x="no_of_cylinders", y ="count", data = temp, kind = "bar", palette="bright")

plt.show()
# Checking the fuel type 

sns.set()

plt.title("The Type of fuel_type in vechicle")

sns.countplot(df.fuel_type, palette="bright")

plt.show()
# Checking the fuel_consumption_city_(l_per_100_km)	 

sns.boxplot(df["fuel_consumption_city_(l_per_100_km)"])

plt.show ()

sns.boxplot(df["fuel_consumption_hwy_(l_per_100_km)"])

plt.show()
# Checking fuel consumption on highway by type



sns.set()

plt.title("The fuel consumption per fuel type")

sns.violinplot(y="fuel_consumption_city_(l_per_100_km)", x = "fuel_type", data= df)

plt.show()
# Checking fuel consumption by type



sns.set()

plt.title("The fuel consumption on highway per fuel type")

sns.boxplot(y="fuel_consumption_hwy_(l_per_100_km)", x = "fuel_type", data= df, palette="bright", sym = "")

plt.show()
# Checking fuel consumption by type



sns.set()

plt.title("The fuel consumption on combination per fuel type")

sns.boxplot(x="fuel_consumption_comb_(l_per_100_km)", y = "fuel_type", data= df, palette="inferno", sym = "")

plt.show()
# checking the fuel consumption and the co2 emission



sns.set()

sns.relplot(x="fuel_consumption_comb_(mpg)", y="co2_emissions(g_per_km)", data = df, palette="bright", hue = "fuel_type", col = "fuel_type", col_wrap=2)

plt.show()
# Checking the  transmission type  

sns.set()

plt.figure(figsize = (10,10))

sns.countplot(y= df["transmission"])

plt.show()
temp1 =  df.groupby("fuel_type")["co2_emissions(g_per_km)"].mean().reset_index()
sns.set()

print("Average CO2 Emission Per fuel type")

g = sns.catplot(x ="fuel_type", y ="co2_emissions(g_per_km)", data = temp1, kind = "bar", palette="icefire")

plt.show()
temp3 = round(df.groupby("vehicle_class")["co2_emissions(g_per_km)"].mean().reset_index(), 2)
sns.set()

print("Average CO2 Emission per vehicle_class ")

g = sns.catplot(y ="vehicle_class", x ="co2_emissions(g_per_km)", data = temp3, kind = "bar", palette="cubehelix_r")

plt.show()