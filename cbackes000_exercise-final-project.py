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

my_filepath = "../input/animal-crossing/items.csv"



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath)



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.head()
# Create a plot

my_data = my_data.loc[my_data.sell_value < 100000]

fig = plt.figure()

rect = 0,0,2,3

log_ax = fig.add_axes(rect)

log_ax.set_yscale("log")

sns.swarmplot(x='orderable',y='sell_value', data=my_data, ax = log_ax, color="black")

sns.boxplot(x='orderable',y='sell_value', data=my_data, ax = log_ax)

#ax.set_yscale('log')





# Your code here





# Check that a figure appears below

step_4.check()