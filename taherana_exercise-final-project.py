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

my_filepath = '../input/pokemon/Pokemon.csv'



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath, index_col="Total", parse_dates=True)



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.head()
my_data.info()
my_data.columns
# Create a plot

 # Your code here

plt.figure(figsize=(50,50))

sns.barplot(x=my_data['HP'], y=my_data.index)

plt.xlabel("total")

plt.title("pokemone digram")

# Check that a figure appears below

step_4.check()
plt.figure(figsize=(20,8))

sns.lineplot(data=my_data['HP'],label="HP")

plt.xlabel("")

plt.title('total power for pokemon')
# Set the width and height of the figure

plt.figure(figsize=(10,6))



# Add title

plt.title("Average of HP, by total")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(x=my_data.index, y=my_data['HP'])



# Add label for vertical axis

plt.ylabel("HP")