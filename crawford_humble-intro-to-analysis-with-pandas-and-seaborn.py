# Numpy is generally used for making fancier lists called arrays and series. 

import numpy as np 



# Pandas is super important, it's the foundation data analysis library we're using.

import pandas as pd 



# Matplotlib is the python plotting library and folks generally import it as "plt"

import matplotlib.pyplot as plt 



# Seaborn is a wrapper for Matplotlib and makes some things easier, generally imported as "sns"

import seaborn as sns 
# Dataset location

database = '../input/database.csv'



# Read in a CSV file and store the contents in a dataframe (df)

df = pd.read_csv(database, low_memory=False)
df.shape
# Head() the dataframe

df.head(4)
# Accessing the features (column names)

df.columns
# Accessing the index (row names)

df.index
# Re-reading the data file to clean up NULL values that make ugly graphs

df = pd.read_csv(database,

                 low_memory=False,             ### Prevents low_memory warning

                 na_values=['UNKNOWN', 'UNK'], ### Adds UNKNOWN and UNK to list of NULLs

                 na_filter=True,               ### Detect NA/NaN as actual NULL values

                 skip_blank_lines=True)        ### Skip boring blank lines

# These are the columns we're going to take from the original dataframe

subset_list = ["Incident Year",

               "Incident Month",

               "Incident Day",

               "Operator",

               "State",

               "Airport", 

               "Flight Phase",

               "Species Name"]



# We're saving them into a new dataframe

df = pd.DataFrame(data=df, columns=subset_list)



# ...dropping NA's

df = df.dropna(thresh=8)



# ...and resetting the index 

df = df.reset_index(drop=True)

df["Operator"].value_counts().head(10)
# Get the numnber of occurances of each operator

operator_counts = df.Operator.value_counts()



# Split and Save the Operator names in a variable

operators = operator_counts.index



# Split and Save the counts in another variable

counts = operator_counts.get_values()
# Create barplot object

barplot = sns.barplot(x=operators, y=counts)
# Create barplot object

barplot = sns.barplot(x=operators[:10], y=counts[:10])
# Create barplot object

plt.xticks(rotation=90)

barplot = sns.barplot(x=operators[:10], y=counts[:10])
# Create and Set a color palette with ridiculous color names

my_palette = ["SlateGray", "CornflowerBlue", "Gold", "SpringGreen"]

current_palette = sns.color_palette(my_palette)

sns.set_palette(current_palette, 10)



# Rotate the x-axis labels

plt.xticks(rotation=90)



# Create the barplot object

barplot = sns.barplot(x=operators[:10], y=counts[:10])
# Create and Set the color palette

paired_palette = sns.color_palette("colorblind")

sns.set_palette(paired_palette, 10)



# Rotate the x-labels

plt.xticks(rotation=45)



# Add the x-axis title

plt.xlabel("x-axis Title: Airline operators", fontsize=20)



# Add the y-axis title

plt.ylabel("y-axis Title: Number of birdstrikes", fontsize=20)



# Add the plot title

plt.title("Main title: Birdstrikes per Airline Operator", fontsize=20)



# Create the plot

barplot = sns.barplot(x=operators[:10], y=counts[:10])