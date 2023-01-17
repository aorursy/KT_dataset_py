import pandas as pd # Datensets

import numpy as np # Data Manipulation

import os # File System

import matplotlib.pyplot as plt # Library for Plotting

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import seaborn as sns # Library for Plotting

sns.set # make plots look nicer

sns.set_palette("husl")

import warnings

warnings.filterwarnings('ignore')

# Plot inside Notebooks

%matplotlib inline 
# Read in Data

data = pd.read_csv("../input/avocado_clean.csv", parse_dates = ['Date'])

data = data[data.region != "TotalUS"]
data.head()
# Create Sample Dataset



testcolumns = ["Date", "Volume", "Name", "Price", "42"]



testdata = [

    ["01-01-2019", 100, "Redhat", 20, 42],    ["01-01-2019", 200, "Pop", 10, 42],    ["01-01-2019", 300, "Mint", 5, 42],    ["01-01-2019", 400, "Arch", 2.5, 42],

    ["01-01-2019", 500, "Suse", 1.25, 42],    ["02-01-2019", 100, "Redhat", 20, 42],    ["02-01-2019", 100, "Pop", 12, 42],    ["02-01-2019", 500, "Mint", 5, 42],

    ["02-01-2019", 400, "Arch", 4.5, 42],    ["02-01-2019", 200, "Suse", 1.25, 42],    ["03-01-2019", 100, "Redhat", 25, 42],    ["03-01-2019", 500, "Pop", 14, 42],

    ["03-01-2019", 300, "Mint", 8, 42],    ["03-01-2019", 200, "Arch", 2.5, 42],    ["03-01-2019", 200, "Suse", 1.25, 42],    ["04-01-2019", 100, "Redhat", 20, 42],

    ["04-01-2019", 600, "Pop", 18, 42],    ["04-01-2019", 100, "Mint", 5, 42],    ["04-01-2019", 100, "Arch", 2.5, 42],    ["04-01-2019", 400, "Suse", 10.25, 42],

    ["05-01-2019", 100, "Redhat", 20, 42],    ["05-01-2019", 800, "Pop", 26, 42],    ["05-01-2019", 500, "Mint", 5, 42],    ["05-01-2019", 200, "Arch", 2.5, 42],

    ["05-01-2019", 500, "Suse", 1.25, 42]

]



testdata = pd.DataFrame(columns = testcolumns, data = testdata)
# Example

fig, ax = plt.subplots(figsize=(15, 5)) # Size of Plot

ax = sns.lineplot(x="Date", y="Volume", data=testdata)

plt.show()
#ToDo: Plot Data by time and price

fig, ax = plt.subplots(figsize=(15, 5)) # Size of Plot

ax = sns.lineplot(x="Date", y="AveragePrice", style = "type", data=data)

plt.show()
#ToDo: Plot Data by time and Volume/Sales

fig, ax = plt.subplots(figsize=(15, 5)) # Size of Plot

ax = sns.lineplot(x="Date", y="Total Volume", hue="type", data=data)

plt.show()
# Example

fig, ax = plt.subplots(figsize=(15, 5)) # Size of Plot

ax = sns.lineplot(x="Date", y="Volume", hue="Name", data=testdata)

plt.show()
#ToDo: Plot Data by time and price and type with style

#ToDo: Plot Data by time and price and type with hue

# Plotting several lines into one plot:

fig, ax = plt.subplots(figsize=(25, 10))

ax = sns.lineplot(x="Date", y="Volume",data=testdata, label="VOL")

ax = sns.lineplot(x="Date", y="Price",data=testdata, label = "PR")

ax = sns.lineplot(x="Date", y="42",data=testdata, label = "42")

plt.legend()

plt.show()
#ToDo: Plot date and volume of S, L and XL Avocados in one graph

#ToDo: Plot date and volume of S, L and XL Bags in one graph

# Splitting Data

redhatdata = testdata[testdata["Name"] == "Redhat"]

popdata = testdata[testdata["Name"] == "Pop"]



# plot first data

fig, ax = plt.subplots(figsize=(10, 5))

ax = sns.lineplot(x="Date", y="Price",data=redhatdata, label="Redhat")

plt.legend()

plt.show()



#plot second data

fig, ax = plt.subplots(figsize=(10, 5))

ax = sns.lineplot(x="Date", y="Price",data=popdata, label="Pop_OS!")

plt.legend()

plt.show()
#ToDo: Plot a graph time and Volume for organic and conventional avocados seperately

#plotting bars

fig, ax = plt.subplots(figsize=(10, 5))

ax = sns.barplot(x="Date", y="Price",hue="Name", data=testdata)

plt.legend()

plt.show()
# ToDo: Plot a barchart with total volume and year, seperate by type

# ToDo: Plot a barchart with type and real price, seperate by year

# sample scatterplot

fig, ax = plt.subplots(figsize=(25, 10))

ax = sns.scatterplot(x="Volume", y="Price", data=testdata)

plt.legend()

plt.show()
# ToDo: Scatter the data by S volume and L Volume

# sample scatterplot

fig, ax = plt.subplots(figsize=(25, 10))

ax = sns.scatterplot(x="Volume", y="Price", hue="Name", size = "Volume", sizes=(10, 400), data=testdata)

plt.legend()

plt.show()
# ToDo: Scatter the data by S and L, seperate by year and AveragePrice

# ToDo: Scatter by S and L, seperate by region and averageprice
# A huge plot: (Double Klick for Zoom)

fig, ax = plt.subplots(figsize=(80, 10))

ax = sns.scatterplot(x="Price", y="Volume",size="Volume", hue="Name", sizes=(10, 400), data=testdata)

plt.legend()

plt.show()
# ToDo: Plot a large graph each for S, L and XL by region. You may seperate

# 42 zu ersetzen:

a1 = [1, "42"] #Ergebnis 1

a2 = [2, 42] #Ergebnis 2

a3 = [3, 42] #Ergebnis 3

a4 = [4, 42] #Ergebnis ...

a5 = [5, "42"]

a6 = [6, "42"]

a7 = [7, 42]

a8 = [8, 42]

a9 = [9, 42]

a10 = [10, 42]

a11 = [11, "42"]



antworten = [a1,a2,a3,a4,a5,a6, a7, a8, a9, a10, a11]

meine_antworten = pd.DataFrame(antworten, columns = ["Id", "Category"])

meine_antworten.to_csv("meine_loesung_Aufgaben2.csv", index = False)