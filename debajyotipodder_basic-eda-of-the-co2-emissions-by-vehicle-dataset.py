import pandas as pd

import numpy as np

from numpy import percentile

from scipy import stats

from scipy.stats import skew

from scipy.special import boxcox1p

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
fuel_con = pd.read_csv("../input/co2-emission-by-vehicles/CO2 Emissions_Canada.csv")
fuel_con.head()
fuel_con.info()
fuel_con.shape
fuel_con.columns
fuel_con.isnull().sum()
fuel_con["Make"].nunique()
fuel_con["Make"].unique()
fuel_con.Model.nunique()
fuel_con["Vehicle Class"].nunique()
fuel_con["Vehicle Class"].unique()
fuel_con["Transmission"].nunique()
fuel_con["Transmission"].unique()
fuel_con["Transmission"] = np.where(fuel_con["Transmission"].isin(["A4", "A5", "A6", "A7", "A8", "A9", "A10"]), "Automatic", fuel_con["Transmission"])

fuel_con["Transmission"] = np.where(fuel_con["Transmission"].isin(["AM5", "AM6", "AM7", "AM8", "AM9"]), "Automated Manual", fuel_con["Transmission"])

fuel_con["Transmission"] = np.where(fuel_con["Transmission"].isin(["AS4", "AS5", "AS6", "AS7", "AS8", "AS9", "AS10"]), "Automatic with Select Shift", fuel_con["Transmission"])

fuel_con["Transmission"] = np.where(fuel_con["Transmission"].isin(["AV", "AV6", "AV7", "AV8", "AV10"]), "Continuously Variable", fuel_con["Transmission"])

fuel_con["Transmission"] = np.where(fuel_con["Transmission"].isin(["M5", "M6", "M7"]), "Manual", fuel_con["Transmission"])
fuel_con["Transmission"].unique()
fuel_con["Fuel Type"].nunique()
fuel_con["Fuel Type"].unique()
fuel_con["Fuel Type"] = np.where(fuel_con["Fuel Type"]=="Z", "Premium Gasoline", fuel_con["Fuel Type"])

fuel_con["Fuel Type"] = np.where(fuel_con["Fuel Type"]=="X", "Regular Gasoline", fuel_con["Fuel Type"])

fuel_con["Fuel Type"] = np.where(fuel_con["Fuel Type"]=="D", "Diesel", fuel_con["Fuel Type"])

fuel_con["Fuel Type"] = np.where(fuel_con["Fuel Type"]=="E", "Ethanol(E85)", fuel_con["Fuel Type"])

fuel_con["Fuel Type"] = np.where(fuel_con["Fuel Type"]=="N", "Natural Gas", fuel_con["Fuel Type"])
fuel_con["Fuel Type"].unique()
fuel_con.head()
from tabulate import tabulate

print("Make")

print(tabulate(pd.DataFrame(fuel_con.Make.value_counts())))



plt.figure(figsize=(19,5));

fuel_con.groupby("Make")["Make"].count().sort_values(ascending=False).plot(kind="bar")

plt.title("Frequency distribution of feature : Make", fontsize=20)

plt.ylabel("Frequency", fontsize=15)

plt.xlabel("Brand Name", fontsize=15)

plt.xticks(rotation=90)

plt.tight_layout()

plt.show()
print(f"Top 20 car models out of total {fuel_con.Model.nunique()} car models")

print(tabulate(pd.DataFrame(fuel_con.Model.value_counts().sort_values(ascending=False)[:20])))



plt.figure(figsize=(19,5));

fuel_con.groupby("Model")["Model"].count().sort_values(ascending=False)[:20].plot(kind="bar")

plt.title("Frequency distribution of feature : Car models (Top 20 plotted)", fontsize=20)

plt.ylabel("Frequency", fontsize=15)

plt.xlabel("Car models", fontsize=15)

plt.xticks(rotation=90)

plt.tight_layout()

plt.show()
print("Vehicle Class")

print(tabulate(pd.DataFrame(fuel_con["Vehicle Class"].value_counts())))



plt.figure(figsize=(19,5));

fuel_con.groupby("Vehicle Class")["Vehicle Class"].count().sort_values(ascending=False).plot(kind="bar")

plt.title("Frequency distribution of feature : Vehicle Class", fontsize=20)

plt.ylabel("Frequency", fontsize=15)

plt.xlabel("Class Type", fontsize=15)

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()
print("Transmission")

print(tabulate(pd.DataFrame(fuel_con.Transmission.value_counts())))



plt.figure(figsize=(15,5));

fuel_con.groupby("Transmission")["Transmission"].count().sort_values(ascending=False).plot(kind="bar")

plt.title("Frequency distribution of feature : Transmission", fontsize=20)

plt.ylabel("Frequency", fontsize=15)

plt.xlabel("Tranmission Type", fontsize=15)

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()
print("Fuel Type")

print(tabulate(pd.DataFrame(fuel_con["Fuel Type"].value_counts())))



plt.figure(figsize=(15,5));

fuel_con.groupby("Fuel Type")["Fuel Type"].count().sort_values(ascending=False).plot(kind="bar")



plt.title("Frequency distribution of feature : Fuel Type", fontsize=20)

plt.ylabel("Frequency", fontsize=15)

plt.xlabel(" Fuel Type", fontsize=15)

plt.xticks(rotation=0)

plt.tight_layout()

plt.show()
CO2_make = fuel_con.groupby(['Make'])['CO2 Emissions(g/km)'].mean().sort_values().reset_index()



plt.figure(figsize=(20,8))

sns.barplot(x = "Make",y="CO2 Emissions(g/km)",data = CO2_make,

            edgecolor=sns.color_palette("dark", 3))

plt.title('CO2 Emissions variation with Brand', fontsize=15)

plt.xlabel('Brand', fontsize=12)

plt.xticks(rotation=90, horizontalalignment='center')

plt.ylabel('CO2 Emissions(g/km)', fontsize=12)
plt.figure(figsize=(16,7))

order = fuel_con.groupby("Make")["CO2 Emissions(g/km)"].median().sort_values(ascending=True).index

sns.boxplot(x="Make", y="CO2 Emissions(g/km)", data=fuel_con, order=order, width=0.5)

plt.title("Distribution of CO2 Emissions in relation to Make", fontsize=15)

plt.xticks(rotation=90, horizontalalignment='center')

plt.xlabel("Make", fontsize=12)

plt.ylabel("CO2 Emissions(g/km)", fontsize=12)

plt.axhline(fuel_con["CO2 Emissions(g/km)"].median(),color='r',linestyle='dashed',linewidth=2)

plt.tight_layout()

plt.show()
CO2_class = fuel_con.groupby(['Vehicle Class'])['CO2 Emissions(g/km)'].mean().sort_values().reset_index()



plt.figure(figsize=(20,8))

sns.barplot(x = "Vehicle Class",y="CO2 Emissions(g/km)",data = CO2_class,

            edgecolor=sns.color_palette("dark", 3))

plt.title('CO2 Emissions variation with Vehicle Class', fontsize=15)

plt.xlabel('Vehicle Class', fontsize=12)

plt.xticks(rotation=90, horizontalalignment='center')

plt.ylabel('CO2 Emissions(g/km)', fontsize=12)
plt.figure(figsize=(16,7))

order = fuel_con.groupby("Vehicle Class")["CO2 Emissions(g/km)"].median().sort_values(ascending=True).index

sns.boxplot(x="Vehicle Class", y="CO2 Emissions(g/km)", data=fuel_con, order=order, width=0.5)

plt.title("Distribution of CO2 Emissions in relation to Vehicle Class", fontsize=15)

plt.xticks(rotation=90, horizontalalignment='center')

plt.xlabel("Vehicle Class", fontsize=12)

plt.ylabel("CO2 Emissions(g/km)", fontsize=12)

plt.axhline(fuel_con["CO2 Emissions(g/km)"].median(),color='r',linestyle='dashed',linewidth=2)

plt.tight_layout()

plt.show()
CO2_transmission = fuel_con.groupby(["Transmission"])['CO2 Emissions(g/km)'].mean().sort_values().reset_index()



plt.figure(figsize=(18,5))

sns.barplot(x = "Transmission",y="CO2 Emissions(g/km)", data = CO2_transmission,

            edgecolor=sns.color_palette("dark", 3))

plt.title('CO2 Emissions variation with Transmission', fontsize=15)

plt.xlabel('\nTransmission', fontsize=12)

plt.xticks(horizontalalignment='center')

plt.ylabel('CO2 Emissions(g/km)', fontsize=12)
plt.figure(figsize=(16,7))

order = fuel_con.groupby("Transmission")["CO2 Emissions(g/km)"].median().sort_values(ascending=True).index

sns.boxplot(x="Transmission", y="CO2 Emissions(g/km)", data=fuel_con, order=order, width=0.5)

plt.title("Distribution of CO2 Emissions in relation to Transmission", fontsize=15)

plt.xlabel("\nTransmission", fontsize=12)

plt.ylabel("CO2 Emissions(g/km)", fontsize=12)

plt.axhline(fuel_con["CO2 Emissions(g/km)"].median(),color='r',linestyle='dashed',linewidth=2)

plt.tight_layout()

plt.show()
CO2_fuel_type = fuel_con.groupby(['Fuel Type'])['CO2 Emissions(g/km)'].mean().sort_values().reset_index()



plt.figure(figsize=(15,5))

sns.barplot(x = "Fuel Type",y="CO2 Emissions(g/km)",data = CO2_fuel_type,

            edgecolor=sns.color_palette("dark", 3))

plt.title('CO2 Emissions variation with Fuel Type', fontsize=15)

plt.xlabel('\nFuel Type', fontsize=12)

plt.ylabel('CO2 Emissions(g/km)', fontsize=12)
plt.figure(figsize=(16,7))

order = fuel_con.groupby("Fuel Type")["CO2 Emissions(g/km)"].median().sort_values(ascending=True).index

sns.boxplot(x="Fuel Type", y="CO2 Emissions(g/km)", data=fuel_con, order=order, width=0.5)

plt.title("Distribution of CO2 Emissions in relation to Fuel Type", fontsize=15)

plt.xticks(rotation=90, horizontalalignment='center')

plt.xlabel("\nFuel", fontsize=12)

plt.ylabel("CO2 Emissions(g/km)", fontsize=12)

plt.axhline(fuel_con["CO2 Emissions(g/km)"].mean(),color='r',linestyle='dashed',linewidth=2)

plt.tight_layout()

plt.show()
stats_ = fuel_con.describe().T.drop(["count"], axis=1)

stats_ = pd.concat([stats_, fuel_con.skew()], axis=1)

stats_.columns = ["mean", "std", "min", "25%", "median", "75%", "max", "skew"]

cols = ["mean", "25%", "median", "75%", "std", "skew", "min", "max"]

stats_ = stats_[cols]

print(tabulate(stats_, headers="keys", floatfmt=".2f"))
for feature in fuel_con.select_dtypes("number").columns:

    

    plt.figure(figsize=(16,5))

    sns.distplot(fuel_con[feature], hist_kws={"rwidth": 0.9})

    plt.xlim(fuel_con[feature].min(), fuel_con[feature].max())

    plt.title(f"Distribution shape of {feature.capitalize()}\n", fontsize=15)

    plt.tight_layout()

    plt.show()
plt.figure(figsize=(16,5))

fuel_con["CO2 Emissions(g/km)"].plot(kind="hist", bins=100, rwidth=0.9)

plt.title("CO2 Emissions(g/km): value distribution")

plt.xlabel("CO2 Emissions(g/km)")

plt.tight_layout()

plt.show()



plt.figure(figsize=(16,5))

fuel_con["CO2 Emissions(g/km)"].plot(kind="box", vert=False)

plt.title("CO2 Emissions(g/km): Frequency distribution\n", fontsize=15)

plt.xlabel("\nCO2 Emissions(g/km)")

plt.yticks([0], [''])

plt.ylabel("CO2 Emissions(g/km)\n", rotation=90)

plt.tight_layout()

plt.show()
corr = fuel_con.corr()

plt.subplots(figsize=(16,16));

sns.heatmap(corr, annot=True, cmap="RdBu", square=True)

plt.title("Correlation matrix of numerical features")

plt.tight_layout()

plt.show()
plt.figure(figsize=(16,5))

corr["CO2 Emissions(g/km)"].sort_values(ascending=True)[:-1].plot(kind="barh")

plt.title("Correlation of numerical features to CO2 Emissions\n", fontsize=15)

plt.xlabel("\nCorrelation to CO2 Emissions")

plt.tight_layout()

plt.show()
CO2_cylinder = fuel_con.groupby(['Cylinders'])['CO2 Emissions(g/km)'].mean().reset_index()



plt.figure(figsize=(15,5))

sns.barplot(x = "Cylinders",y="CO2 Emissions(g/km)",data = CO2_cylinder,

            edgecolor=sns.color_palette("dark", 3))

plt.title('CO2 Emissions with number of Cylinders\n', fontsize=15)

plt.xlabel('Cylinders', fontsize=12)

plt.ylabel('CO2 Emissions(g/km)', fontsize=12)
fuel_cylinder = fuel_con.groupby(['Cylinders'])['Fuel Consumption Comb (L/100 km)'].mean().reset_index()



plt.figure(figsize=(15,5))

sns.barplot(x = "Cylinders",y="Fuel Consumption Comb (L/100 km)",data = fuel_cylinder,

            edgecolor=sns.color_palette("dark", 3))

plt.title('Fuel Consumption with number of Cylinders\n', fontsize=15)

plt.xlabel('Cylinders', fontsize=12)

plt.ylabel('Fuel Consumption Comb (L/100 km)', fontsize=12)
plt.figure(figsize=(16,7))

order = fuel_con.groupby("Cylinders")["CO2 Emissions(g/km)"].median().sort_values(ascending=True).index

sns.boxplot(x="Cylinders", y="CO2 Emissions(g/km)", data=fuel_con, order=order, width=0.5)

plt.title("Distribution of CO2 Emissions in relation to number of Cylinders", fontsize=15)

plt.xlabel("Cylinders", fontsize=12)

plt.ylabel("CO2 Emissions(g/km)", fontsize=12)

plt.axhline(fuel_con["CO2 Emissions(g/km)"].median(),color='r',linestyle='dashed',linewidth=2)

plt.tight_layout()

plt.show()
CO2_engine = fuel_con.groupby(['Engine Size(L)'])['CO2 Emissions(g/km)'].mean().reset_index()



plt.figure(figsize=(18,8))

sns.barplot(x = "Engine Size(L)",y="CO2 Emissions(g/km)",data = CO2_engine,

            edgecolor=sns.color_palette("dark", 3))

plt.title('CO2 Emissions with Engine Size', fontsize=15)

plt.xlabel('Engine Size', fontsize=12)

plt.ylabel('CO2 Emissions(g/km)', fontsize=12)
fuel_engine = fuel_con.groupby(['Engine Size(L)'])['Fuel Consumption Comb (L/100 km)'].mean().reset_index()



plt.figure(figsize=(20,8))

sns.barplot(x = "Engine Size(L)",y="Fuel Consumption Comb (L/100 km)",data = fuel_engine,

            edgecolor=sns.color_palette("dark", 3))

plt.title('Fuel Consumption with Engine Size(L)\n', fontsize=15)

plt.xlabel('Engine Size(L)', fontsize=12)

plt.ylabel('Fuel Consumption Comb (L/100 km)', fontsize=12)
plt.figure(figsize=(16,7))

order = fuel_con.groupby("Engine Size(L)")["CO2 Emissions(g/km)"].median().index

sns.boxplot(x="Engine Size(L)", y="CO2 Emissions(g/km)", data=fuel_con, order=order, width=0.5)

plt.title("Distribution of CO2 Emissions in relation to Engine Size(L)", fontsize = 15)

plt.xlabel("Engine Size(L)", fontsize = 12)

plt.ylabel("CO2 Emissions(g/km)", fontsize=12)

plt.axhline(fuel_con["CO2 Emissions(g/km)"].median(),color='r',linestyle='dashed',linewidth=2)

plt.tight_layout()

plt.show()
CO2_city = fuel_con.groupby(['Fuel Consumption City (L/100 km)'])['CO2 Emissions(g/km)'].mean().reset_index()



plt.figure(figsize=(25,8))

sns.barplot(x = "Fuel Consumption City (L/100 km)", y="CO2 Emissions(g/km)",data = CO2_city,

            edgecolor=sns.color_palette("dark", 3))

plt.title('CO2 Emissions with Fuel Consumption City (L/100 km)', fontsize=15)

plt.xlabel('Fuel Consumption City (L/100 km)', fontsize=12)

plt.xticks(rotation=90, horizontalalignment='center', fontweight='light', fontsize='7')

plt.ylabel('CO2 Emissions(g/km)', fontsize=12)
CO2_hwy = fuel_con.groupby(['Fuel Consumption Hwy (L/100 km)'])['CO2 Emissions(g/km)'].mean().reset_index()



plt.figure(figsize=(25,8))

sns.barplot(x = "Fuel Consumption Hwy (L/100 km)", y="CO2 Emissions(g/km)",data = CO2_hwy,

            edgecolor=sns.color_palette("dark", 3))

plt.title('CO2 Emissions with Fuel Consumption Hwy (L/100 km)', fontsize=15)

plt.xlabel('Fuel Consumption Hwy (L/100 km)', fontsize=12)

plt.xticks(rotation=90, horizontalalignment='center', fontweight='light', fontsize='7')

plt.ylabel('CO2 Emissions(g/km)', fontsize=12)
CO2_comb = fuel_con.groupby(['Fuel Consumption Comb (L/100 km)'])['CO2 Emissions(g/km)'].mean().reset_index()



plt.figure(figsize=(25,8))

sns.barplot(x = "Fuel Consumption Comb (L/100 km)", y="CO2 Emissions(g/km)",data = CO2_comb,

            edgecolor=sns.color_palette("dark", 3))

plt.title('CO2 Emissions with Fuel Consumption Comb (L/100 km)', fontsize=15)

plt.xlabel('Fuel Consumption Comb (L/100 km)', fontsize=12)

plt.xticks(rotation=90, horizontalalignment='center', fontweight='light', fontsize='7')

plt.ylabel('CO2 Emissions(g/km)', fontsize=12)
CO2_comb_mpg = fuel_con.groupby(['Fuel Consumption Comb (mpg)'])['CO2 Emissions(g/km)'].mean().reset_index()



plt.figure(figsize=(25,8))

sns.barplot(x = "Fuel Consumption Comb (mpg)", y="CO2 Emissions(g/km)",data = CO2_comb_mpg,

            edgecolor=sns.color_palette("dark", 3))

plt.title('CO2 Emissions with Fuel Consumption Comb (mpg)', fontsize=15)

plt.xlabel('Fuel Consumption Comb (mpg)', fontsize=12)

plt.xticks(rotation=90, horizontalalignment='center', fontweight='light', fontsize='12')

plt.ylabel('CO2 Emissions(g/km)', fontsize=12)