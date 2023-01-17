import pandas as pd

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

my_filepath = "../input/nys-environmental-remediation-sites/environmental-remediation-sites.csv"



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath)

my_data = my_data.dropna(axis=1)

print(my_data.columns , my_data.shape)



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.head()

#print(my_data.shape)
my_data = my_data.dropna(axis=0)

my_data.shape[:100]
# Create a plot



# Set the width and height of the figure

plt.figure(figsize=(10,6))



# Add title

#plt.title("")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(x=my_data['Program Number'], y=my_data['SWIS Code'])



# Add label for vertical axis

#plt.ylabel("")# Your code here



# Check that a figure appears below

step_4.check()