import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

import os

if not os.path.exists("../input/ign_scores.csv"):

    os.symlink("../input/data-for-datavis/ign_scores.csv", "../input/ign_scores.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex3 import *

print("Setup Complete")
# Path of the file to read

ign_filepath = "../input/ign_scores.csv"



# Fill in the line below to read the file into a variable ign_data

ign_data = pd.read_csv(ign_filepath, index_col="Platform")



# Run the line below with no changes to check that you've loaded the data correctly

step_1.check()
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
# Print the data

ign_data
# Fill in the line below: What is the highest average score received by PC games,

# for any platform?

high_score =  7.759930



# Fill in the line below: On the Playstation Vita platform, which genre has the 

# lowest average score? Please provide the name of the column, and put your answer 

# in single quotes (e.g., 'Action', 'Adventure', 'Fighting', etc.)

worst_genre = 'Simulation'



# Check your answers

step_2.check()
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
# Bar chart showing average score for racing games by platform

# Set the width and height of the figure

plt.figure(figsize=(16,6))



# Add title

plt.title("Average score for racing games, for each platform")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(x=ign_data['Racing'], y=ign_data.index)



# Add label for vertical axis

plt.ylabel("Plateform")

# Add label for horizontal axis

plt.xlabel("")



# Check your answer

step_3.a.check()
# Lines below will give you a hint or solution code

#step_3.a.hint()

#step_3.a.solution_plot()
#step_3.b.hint()
# Check your answer (Run this code cell to receive credit!)

step_3.b.solution()
# Heatmap showing average game score by platform and genre

# Set the width and height of the figure

plt.figure(figsize=(14,7))



# Add title

plt.title("Average score for genre, by platform")



# Heatmap showing average arrival delay for each airline by month

sns.heatmap(data=ign_data, annot=True)



# Add label for horizontal axis

plt.xlabel("genre")



# Check your answer

step_4.a.check()
# Lines below will give you a hint or solution code

#step_4.a.hint()

#step_4.a.solution_plot()
#step_4.b.hint()
# Check your answer (Run this code cell to receive credit!)

step_4.b.solution()