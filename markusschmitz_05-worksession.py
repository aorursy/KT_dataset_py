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

#ToDo: Plot Data by time and Volume/Sales

# sample solution:

fig, ax = plt.subplots(figsize=(25, 10)) # Size of Plot

ax = sns.lineplot(x="Date", y="AveragePrice", data=data)

plt.show()
#sample solution

fig, ax = plt.subplots(figsize=(25, 10)) # Size of Plot

ax = sns.lineplot(x="Date", y="Total Volume", data=data)

plt.show()
# Example:

# Example

fig, ax = plt.subplots(figsize=(15, 5)) # Size of Plot

ax = sns.lineplot(x="Date", y="Volume", hue="Name", data=testdata)

plt.show()
#ToDo: Plot Data by time and price and type with style

#ToDo: Plot Data by time and price and type with hue

#sample Solution

fig, ax = plt.subplots(figsize=(25, 10))

sns.lineplot(x="Date", y="AveragePrice",style="type",data=data)

plt.show()
#sample Solution

fig, ax = plt.subplots(figsize=(25, 10))

sns.lineplot(x="Date", y="AveragePrice",hue="type",data=data)

plt.show()
#ToDo: Plot the Avcoado Price andy seperate by type

# Plotting several lines into one plot:

fig, ax = plt.subplots(figsize=(25, 10))

ax = sns.lineplot(x="Date", y="Volume",data=testdata, label="VOL")

ax = sns.lineplot(x="Date", y="Price",data=testdata, label = "PR")

ax = sns.lineplot(x="Date", y="42",data=testdata, label = "42")

plt.legend()

plt.show()
#ToDo: Plot date and volume of S, L and XL Avocados in one graph

#ToDo: Plot date and volume of S, L and XL Bags in one graph

# Sample Solution

fig, ax = plt.subplots(figsize=(25, 10))

ax = sns.lineplot(x="Date", y="S",data=data, label = "S")

ax = sns.lineplot(x="Date", y="L",data=data, label = "L")

ax = sns.lineplot(x="Date", y="XL",data=data, label = "XL")

plt.legend()
# Samplw Solution:

fig, ax = plt.subplots(figsize=(25, 10))

sns.lineplot(x="Date", y="Small Bags", data=data, label = "Bags S")

sns.lineplot(x="Date", y="Large Bags", data=data, label = "Bags L")

sns.lineplot(x="Date", y="XLarge Bags", data=data, label = "Bags XL")

plt.legend()

plt.show()
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

organic = data[data["type"] == "organic"]

conventional = data[data["type"] == "conventional"]



# Sample Solution

fig, ax = plt.subplots(figsize=(25, 10))

ax = sns.lineplot(x="Date", y="S",data=organic, label = "organic S")

ax = sns.lineplot(x="Date", y="L",data=organic, label = "organic L")

ax = sns.lineplot(x="Date", y="XL",data=organic, label = "organic XL")

plt.legend()

plt.show()



fig, ax = plt.subplots(figsize=(25, 10))

ax = sns.lineplot(x="Date", y="S",data=conventional, label = "S")

ax = sns.lineplot(x="Date", y="L",data=conventional, label = "L")

ax = sns.lineplot(x="Date", y="XL",data=conventional, label = "XL")

plt.legend()

plt.show()
#plotting bars

fig, ax = plt.subplots(figsize=(10, 5))

ax = sns.barplot(x="Date", y="Price",hue="Name", data=testdata)

plt.legend()

plt.show()
# ToDo: Plot a barchart with total volume and year, seperate by type

# ToDo: Plot a barchart with type and real price, seperate by year

# sample Solution

fig, ax = plt.subplots(figsize=(25, 10))

ax = sns.barplot(x="year", y="Total Volume",hue="type", data=data)

plt.legend()

plt.show()
# sample Solution

fig, ax = plt.subplots(figsize=(25, 10))

ax = sns.barplot(x="type", y="real Price", hue = "year", data=data)

plt.legend()

plt.show()
# sample scatterplot

fig, ax = plt.subplots(figsize=(25, 10))

ax = sns.scatterplot(x="Volume", y="Price", data=testdata)

plt.legend()

plt.show()
# ToDo: Scatter the data by S volume and L Volume

# sample Solution

fig, ax = plt.subplots(figsize=(25, 10))

ax = sns.scatterplot(x="S", y="L", data=data, label="S/L")

plt.legend()

plt.show()
# sample scatterplot

fig, ax = plt.subplots(figsize=(25, 10))

ax = sns.scatterplot(x="Volume", y="Price", hue="Name", size = "Volume", sizes=(10, 400), data=testdata)

plt.legend()

plt.show()
# ToDo: Scatter the data by S and L, seperate by year and AveragePrice

# ToDo: Scatter by S and L, seperate by region and averageprice
# sample Solution

fig, ax = plt.subplots(figsize=(25, 10))

ax = sns.scatterplot(x="S", y="L", hue = "year",size="AveragePrice", sizes=(10, 400), data=data)

plt.legend()

plt.show()
# sample Solution

fig, ax = plt.subplots(figsize=(25, 25))

ax = sns.scatterplot(x="S", y="L", hue = "region",size="AveragePrice", sizes=(10, 400), data=data)

plt.legend()

plt.show()
# A huge plot: (Double Klick for Zoom)

fig, ax = plt.subplots(figsize=(80, 10))

ax = sns.scatterplot(x="Price", y="Volume",size="Volume", hue="Name", sizes=(10, 400), data=testdata)

plt.legend()

plt.show()
# ToDo: Plot a large graph each for S, L and XL by region. You may seperate

# sample Solution

fig, ax = plt.subplots(figsize=(80, 10))

ax = sns.scatterplot(x="region", y="L",size="L", hue="AveragePrice", sizes=(10, 400), data=data)

plt.legend()

plt.show()

fig, ax = plt.subplots(figsize=(80, 10))

ax = sns.scatterplot(x="region", y="S",size="S", hue="AveragePrice", sizes=(10, 400), data=data)

plt.legend()

plt.show()

fig, ax = plt.subplots(figsize=(80, 10))

ax = sns.scatterplot(x="region", y="XL",size="XL", hue="AveragePrice", sizes=(10, 400), data=data)

plt.legend()

plt.show()

# 42 zu ersetzen:

a1 = [1, "09"] #Ergebnis 1

a2 = [2, 3000000] #Ergebnis 2

a3 = [3, 2017] #Ergebnis 3

a4 = [4, 1.2] #Ergebnis ...

a5 = [5, "False"]

a6 = [6, "L"]

a7 = [7, 2018]

a8 = [8, 2018]

a9 = [9, 4]

a10 = [10, 2015]

a11 = [11, "Los Angeles"]

antworten = [a1,a2,a3,a4,a5,a6, a7, a8, a9, a10, a11]

meine_antworten = pd.DataFrame(antworten, columns = ["Id", "Category"])

meine_antworten.to_csv("meine_loesung_Aufgaben2.csv", index = False)