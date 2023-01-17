import pandas as pd # Datensets
import numpy as np # Data Manipulation
import os # File System
from IPython.display import Image
from IPython.core.display import HTML 
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
data = pd.read_csv("../input/testsets/avocado.csv")
# ToDo: Take a look at the data. 
# Try printing some lines or use describe to get a feel for it
data.head()
# ToDo: Delete all data where region is in statelist
statelist = [ "TotalUS", 'California', 'GreatLakes', 'Midsouth', 'Northeast', 'SouthCarolina', 'SouthCentral', 'Southeast', 'Philadelphia', 'Plains', 'West']
data = data[~data.region.isin(statelist)]
# ToDo: Remove all missing Data from the dataset
data = data.dropna(how="any")
# ToDo: Convert the "Date" column to datetime
data.Date = pd.to_datetime(data.Date)
# ToDo: Calculate RealPrice
data["RealPrice"] = ((2018 - data.Date.dt.year) * 0.02015 + 1) * data["AveragePrice"]
#ToDo: Find the number of remaining regions
len(data.region.unique())
#ToDo: Plot Data by time and price
fig, ax = plt.subplots(figsize=(25, 10)) # Size of Plot
ax = sns.lineplot(x="Date", y="AveragePrice", data=data) # Makes a LinePlot
plt.show()
#ToDo: Plot Data by time and Volume/Sales
fig, ax = plt.subplots(figsize=(25, 10)) # Size of Plot
ax = sns.lineplot(x="Date", y="Total Bags", data=data) # Makes a LinePlot
plt.show()
#ToDo: Plot Data by time and price and type with style
fig, ax = plt.subplots(figsize=(25, 10)) # Size of Plot
ax = sns.lineplot(x="Date", y="AveragePrice", hue="type", data=data)
plt.show()
# ToDo: create subset of Data and cut between 2017-01-01 and 2017-04-15
# ToDo: Define the begin of the cut
begin = pd.to_datetime("2017-01-01")
# ToDo: Define the end of the cut
end = pd.to_datetime("2017-04-15")
# ToDo: Cut the data
subset = data[(data["Date"] > begin) & (data["Date"] < end)]
# ToDo: Plot Data by time and price and type with hue
fig, ax = plt.subplots(figsize=(25, 10)) # Size of Plot
ax = sns.lineplot(x="Date", y="AveragePrice", hue="type", data=subset)
plt.show()
#ToDo: Plot date and volume of S, L and XL Avocados in one graph
fig, ax = plt.subplots(figsize=(25, 10))
ax = sns.lineplot(x="Date", y="S",data=data, label="S")
ax = sns.lineplot(x="Date", y="L",data=data, label = "L")
ax = sns.lineplot(x="Date", y="XL",data=data, label = "XL")
plt.legend()
plt.show()
#ToDo: Plot date and volume of S, L and XL Bags in one graph
fig, ax = plt.subplots(figsize=(25, 10))
ax = sns.lineplot(x="Date", y="Small Bags",data=data, label="S")
ax = sns.lineplot(x="Date", y="Large Bags",data=data, label = "L")
ax = sns.lineplot(x="Date", y="XLarge Bags",data=data, label = "XL")
plt.legend()
plt.show()
#ToDo: Plot a graph time and Volume for organic and conventional avocados seperately
# Splitting Data
conventional = data[data["type"] == "conventional"]
organic = data[data["type"] == "organic"]

# plot first data
fig, ax = plt.subplots(figsize=(25, 10))
ax = sns.lineplot(x="Date", y="Total Volume",data=conventional, label="conventional")
plt.legend()
plt.show()
#plot second data
fig, ax = plt.subplots(figsize=(25, 10))
ax = sns.lineplot(x="Date", y="Total Volume",data=organic, label="organic")
plt.legend()
plt.show()
# ToDo: Calculate sum of Total Volume for organic and conventional
organic_sales = organic["Total Volume"].sum()
conv_sales = conventional["Total Volume"].sum()

# ToDo: Calculate share of organic avocado Sales from all sold avocados
organic_sales / (conv_sales + organic_sales)
# ToDo: Plot a barchart with total volume and year, seperate by type
fig, ax = plt.subplots(figsize=(25, 10))
ax = sns.barplot(x="year", y="Total Volume",data=data, hue = "type")
plt.legend()
plt.show()
# ToDo: Plot a barchart with type and real price, seperate by year
fig, ax = plt.subplots(figsize=(25, 10))
ax = sns.barplot(x="type", y="RealPrice",data=data, hue = "year")
plt.legend()
plt.show()
# ToDo: Scatter the data by S volume and L Volume
fig, ax = plt.subplots(figsize=(25, 10))
ax = sns.scatterplot(x="S", y="L", data=conventional)
plt.legend()
plt.show()
# ToDo: Scatter the data by S volume and L Volume
fig, ax = plt.subplots(figsize=(25, 10))
ax = sns.scatterplot(x="S", y="L", data=conventional[conventional["region"] == "Boston"])
plt.legend()
plt.show()
sns.factorplot(x='AveragePrice',y='region',data=data[data.type=="organic"],hue='year', size = 15, join = False)
# durch Lösungen ersetzen:

a1 = [1, 43]               
a2 = [2, 2017]               
a3 = [3, 2018]               
a4 = [4, 2017]              
a5 = [5, 1.4]    
a6 = [6, "False"]  
a7 = [7, "True"]              
a8 = [8, "Bio"]              
a9 = [9, 0.028]             
a10 = [10, 2018]            
a11 = [11, 2018]  
a12 = [12, "False"]
a13 = [13, 7]
a14 = [14, 3]
a15 = [15, ["HartfordSpringfield", "RaleighGreensboro", "SanFrancisco"]]
antworten = [a1,a2,a3,a4,a5,a6, a7, a8, a9, a10, a11, a12, a13, a14, a15]
meine_antworten = pd.DataFrame(antworten, columns = ["Id", "Solution"])
meine_antworten.to_csv("meine_loesung_Aufgaben2.csv", index = False)
