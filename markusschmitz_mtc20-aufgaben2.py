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
# ToDo: Delete all data where region is in statelist



# ToDo: Remove all missing Data from the dataset



# ToDo: Convert the "Date" column to datetime



# ToDo: Calculate RealPrice



#ToDo: Find the number of remaining regions

#ToDo: Plot Data by time and price



#ToDo: Plot Data by time and Volume/Sales

#ToDo: Plot Data by time and price and type with style

# ToDo: create subset of Data and cut between 2017-01-01 and 2017-04-15

# ToDo: Define the begin of the cut



# ToDo: Define the end of the cut



# ToDo: Cut the data



# ToDo: Plot Data by time and price and type with hue

#ToDo: Plot date and volume of S, L and XL Avocados in one graph

#ToDo: Plot date and volume of S, L and XL Bags in one graph

#ToDo: Plot a graph time and Volume for organic and conventional avocados seperately

# Splitting Data



# plot first data



#plot second data

# ToDo: Calculate sum of Total Volume for organic and conventional



# ToDo: Calculate share of organic avocado Sales from all sold avocados

# ToDo: Plot a barchart with total volume and year, seperate by type

# ToDo: Plot a barchart with type and real price, seperate by year

# ToDo: Scatter the data by S volume and L Volume

# ToDo: Scatter the data by S volume and L Volume

# make a Factorplot to find the regions with highest price
# durch Lösungen ersetzen:



a1 = [1, 0]               

a2 = [2, 0]               

a3 = [3, 0]               

a4 = [4, 0]              

a5 = [5, 0.0]    

a6 = [6, "0"]  

a7 = [7, "0"]              

a8 = [8, "0"]              

a9 = [9, 0.0]             

a10 = [10, 0]            

a11 = [11, 0]  

a12 = [12, "0"]

a13 = [13, 0]

a14 = [14, 0]

a15 = [15, ["0", "0", "0"]]

antworten = [a1,a2,a3,a4,a5,a6, a7, a8, a9, a10, a11, a12, a13, a14, a15]

meine_antworten = pd.DataFrame(antworten, columns = ["Id", "Solution"])

meine_antworten.to_csv("meine_loesung_Aufgaben2.csv", index = False)