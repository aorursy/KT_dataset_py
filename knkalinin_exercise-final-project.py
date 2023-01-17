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

my_filepath_dc = "../input/dc-wikia-data.csv"

my_filepath_ml = "../input/marvel-wikia-data.csv"



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data_dc = pd.read_csv(my_filepath_dc, na_values='0',usecols=[1,3,4,5,6,7,9,10,12],

                         #index_col='name' 

                        )

my_data_ml = pd.read_csv(my_filepath_ml)



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data_dc.head()
my_data_dc.describe(include='all')
#my_data_ml.head(5)
plt.figure(figsize=(18,8))

sns.scatterplot(x='YEAR', 

                y='APPEARANCES',

                hue='ID',

                data=my_data_dc)

                           
# Create a plot

# Your code here

# Set the width and height of the figure

plt.figure(figsize=(18,18))



# Add title

plt.title("Popularity,APPEARANCES, by name ")



# Bar chart showing poularity

sns.barplot(x=my_data_dc.APPEARANCES,y=my_data_dc.name)







# Add label for vertical axis

plt.ylabel("Name")

# Check that a figure appears below

#step_4.check()