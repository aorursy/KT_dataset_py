import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

import os

if not os.path.exists("../input/museum_visitors.csv"):

    os.symlink("../input/data-for-datavis/museum_visitors.csv", "../input/museum_visitors.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex2 import *

print("Setup Complete")
# Path of the file to read

museum_filepath = "../input/museum_visitors.csv"



# Fill in the line below to read the file into a variable museum_data

museum_data = pd.read_csv(museum_filepath, index_col="Date", parse_dates=True)



# Run the line below with no changes to check that you've loaded the data correctly

step_1.check()
# Uncomment the line below to receive a hint

#step_1.hint()

# Uncomment the line below to see the solution

#step_1.solution()
# Print the last five rows of the data 

museum_data.tail() # Your code here
# Fill in the line below: How many visitors did the Chinese American Museum 

# receive in July 2018?

ca_museum_jul18 = 2620 



# Fill in the line below: In October 2018, how many more visitors did Avila 

# Adobe receive than the Firehouse Museum?

avila_oct18 = 19280 - 4622



# Check your answers

step_2.check()
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
# Line chart showing the number of visitors to each museum over time

sns.lineplot(data=museum_data) # Your code here



# Check your answer

step_3.check()
# Lines below will give you a hint or solution code

#step_3.hint()

#step_3.solution_plot()
# Line plot showing the number of visitors to Avila Adobe over time

sns.lineplot(data=museum_data['Avila Adobe'], label="Avila Adobe") # Your code here



# Check your answer

step_4.a.check()
# Lines below will give you a hint or solution code

#step_4.a.hint()

#step_4.a.solution_plot()
step_4.b.hint()
# Check your answer (Run this code cell to receive credit!)

step_4.b.solution()