import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex2 import *

print("Setup Complete")
import pandas as pd

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

____ # Your code here

museum_data.tail()
# Fill in the line below: How many visitors did the Chinese American Museum 

# receive in July 2018?

ca_museum_jul18 = 2620 



# Fill in the line below: In October 2018, how many more visitors did Avila 

# Adobe receive than the Firehouse Museum?

avila_oct18 = 14658



# Check your answers

step_2.check()
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
# Line chart showing the number of visitors to each museum over time

____ # Your code here

plt.figure(figsize=(16,6))

sns.lineplot(data=museum_data)



# Check your answer

step_3.check()
# Lines below will give you a hint or solution code

#step_3.hint()

#step_3.solution_plot()
# Line plot showing the number of visitors to Avila Adobe over time

____ # Your code here



# Set the width and height of the figure

plt.figure(figsize=(14,6))



# Add title

plt.title("Number of visitors to Avila Adobe")



# Line chart showing daily global streams of 'Shape of You'

sns.lineplot(data=museum_data['Avila Adobe'], label="Avila Adobe")



# Add label for horizontal axis

plt.xlabel("Date")



# Check your answer

step_4.a.check()
# Lines below will give you a hint or solution code

#step_4.a.hint()

#step_4.a.solution_plot()
#step_4.b.hint()
#step_4.b.solution()