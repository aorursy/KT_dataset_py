import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex7 import *

print("Setup Complete")
# Check for a dataset with a CSV file

step_1.check()
# Fill in the line below: Specify the path of the CSV file to read

my_filepath = "../input/fivethirtyeight-comic-characters-dataset/dc-wikia-data.csv"



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath,index_col="page_id",parse_dates=["FIRST APPEARANCE"])



# Check that a dataset has been uploaded into my_data

step_3.check()
my_data['FIRST APPEARANCE']= pd.to_datetime(my_data['FIRST APPEARANCE'],errors='coerce') 
# Print the first five rows of the data

my_data.head()
#Get the top 10 highest appearing characters

highest_appearances=my_data.sort_values(by="APPEARANCES",ascending=False).head(10)

highest_appearances
# Create a plot for the highest appearing characters

# Set the width and height of the figure

plt.figure(figsize=(10,6))



# Add title

plt.title("Highest Appearing characters of DC")



# Bar chart 

sns.barplot(y=highest_appearances['name'], x=highest_appearances['APPEARANCES'])



# Add label for vertical axis

plt.xlabel("No of Appearances")

plt.ylabel("Characters")

#plt.xticks(rotation=90)

# Check that a figure appears below

step_4.check()


# Create a Distribution plot for Appearances (all data)

# Set the width and height of the figure

plt.figure(figsize=(10,6))



# Add title

plt.title("Appearance Distribution")



# Histogram showing appearance distribution

sns.distplot(a=my_data['APPEARANCES'], kde=True)
#Now I will remove the top 2 highest appearing characters and compare the KDE with the previous plot.

exluding_top_2=my_data.sort_values(by="APPEARANCES",ascending=False).tail(6894)

exluding_top_2.head()
# Set the width and height of the figure

plt.figure(figsize=(15,6))

# KDE plots , one with all data and the other excluding the top 2 highest appearing characters

sns.kdeplot(data=my_data['APPEARANCES'], label="all-data", shade=True)

sns.kdeplot(data=exluding_top_2['APPEARANCES'], label="excluding-top-2-appearing-characters", shade=True)



# Add title

plt.title("Appearances' Distribution ")