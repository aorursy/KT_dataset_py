import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# set up code checking

import os

if not os.path.exists("../input/ign_scores.csv"):

    os.symlink("../input/data-for-datavis/ign_scores.csv", "../input/ign_scores.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex3 import *

print("Setup Complete")
# path of the file to read

ign_filepath = "../input/ign_scores.csv"



# read the file into a variable ign_data

ign_data = pd.read_csv(ign_filepath, index_col="Platform")



# run the line below with no changes to check that you've loaded the data correctly

step_1.check()
# Lines below will give you a hint or solution code

# step_1.hint()

# step_1.solution()
# Print the data

ign_data
# what is the highest average score received by PC games, for any platform?

high_score = ign_data.loc['PC'].max()



# on the Playstation Vita platform, which genre has the lowest average score?

worst_genre = ign_data.loc['PlayStation Vita'].idxmin()



# check your answers

step_2.check()
# lines below will give you a hint or solution code

# step_2.hint()

# step_2.solution()
# Bar chart showing average score for racing games by platform

plt.figure(figsize=(11.5,5))

plt.xticks(rotation='vertical')

# plt.xticks(rotation=45, horizontalalignment='right')

sns.barplot(x=ign_data.index, y=ign_data['Racing'])



# Check your answer

step_3.a.check()
# lines below will give you a hint or solution code

# step_3.a.hint()

# step_3.a.solution_plot()
#step_3.b.hint()
# check your answer (Run this code cell to receive credit!)

step_3.b.solution()
# heatmap showing average game score by platform and genre

plt.figure(figsize=(12,11.5))

sns.heatmap(data=ign_data, annot=True)



# check your answer

step_4.a.check()
# lines below will give you a hint or solution code

# step_4.a.hint()

# step_4.a.solution_plot()
#step_4.b.hint()
# Check your answer (Run this code cell to receive credit!)

step_4.b.solution()
flight_data = pd.read_csv("../input/flight_delays.csv", index_col="Month")
plt.figure(figsize=(14,6))

plt.title("Average Arrival Delay for Each Airline, by Month")

sns.heatmap(data=flight_data, annot=True)

plt.xlabel("Airline")
# on March, the maximum average delay

print(flight_data.loc[3].max())



# on October, the aireline with the minimum average delay

print(flight_data.loc[10].idxmin())