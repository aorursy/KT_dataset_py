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

my_filepath = "../input/fivethirtyeight-comic-characters-dataset/dc-wikia-data.csv"

flight_delays_filepath = "../input/data-for-datavis/flight_delays.csv"

# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath)

flight_delays_data = pd.read_csv(flight_delays_filepath)

# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.head()
# Create a plot

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set_style("ticks")



# Line chart 

plt.figure(figsize=(12,6))

sns.lineplot(data=flight_delays_data)



# Check that a figure appears below

step_4.check()