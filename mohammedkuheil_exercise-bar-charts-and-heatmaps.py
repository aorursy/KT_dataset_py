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

#print(ign_data)

ign_data = pd.read_csv(ign_filepath, index_col='Platform')



# Run the line below with no changes to check that you've loaded the data correctly

step_1.check()
ign_data
ign_data.shape
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
# Print the data

ign_data.head(30) # Your code here
ign_data.describe()
# Fill in the line below: What is the highest average score received by PC games,

# for any platform?

high_score = max (ign_data.loc['PC'])



# Fill in the line below: On the Playstation Vita platform, which genre has the 

# lowest average score? Please provide the name of the column, and put your answer 

# in single quotes (e.g., 'Action', 'Adventure', 'Fighting', etc.)

worst_genre = ign_data.loc['PlayStation Vita'].idxmin()



# Check your answers

step_2.check()
[ign_data.loc['PlayStation Vita'].idxmin(), ign_data.loc['PlayStation Vita'].min()]
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
# Bar chart showing average score for racing games by platform

plt.figure(figsize=(28,7))

plt.title("Platforms")

sns.barplot(x=ign_data.index, y=ign_data['Racing'])

plt.ylabel("Racing")

plt.xlabel("Platform")

# Check your answer

step_3.a.check()
ign_data.loc['Xbox One'].idxmax()
# Lines below will give you a hint or solution code

#step_3.a.hint()

#step_3.a.solution_plot()
ign_data.Racing.idxmax()
#step_3.b.hint()
# Check your answer (Run this code cell to receive credit!)

step_3.b.solution()
# Heatmap showing average game score by platform and genre

plt.figure(figsize=(16,7))

plt.title("HeatMap")

sns.heatmap(data=ign_data, annot=True)

plt.xlabel("Genre")



# Check your answer

step_4.a.check()
ign_data.max()
# Heatmap showing average game score by platform and genre

plt.figure(figsize=(16,7))

plt.title("HeatMap")

sns.heatmap(data=ign_data[['Action, Adventure','RPG']],cmap="YlGnBu", annot=True)

plt.xlabel("Genre")

ign_data.RPG.mean()
ign_data['Action, Adventure'].mean()
ign_data.index
plt.figure(figsize=(16,7))

plt.title("HeatMap")

sns.heatmap(data=ign_data.loc[['Xbox One', 'PlayStation 4','Dreamcast']],cmap="YlGnBu", annot=True)

plt.xlabel("Genre")
platform_list = ['Xbox One', 'PlayStation 4','Dreamcast']

print (platform_list)

scors = [ign_data.loc[platform].mean() for platform in platform_list]

scors
ign_data.loc[['Xbox One', 'PlayStation 4','Xbox One']].Simulation.mean()
plt.figure(figsize=(16,7))

plt.title("HeatMap")

sns.heatmap(data=ign_data[['RPG', 'Adventure']].loc[['Xbox One', 'PlayStation 4','Xbox One']],cmap="YlGnBu", annot=True)

plt.xlabel("Genre")
# Lines below will give you a hint or solution code

#step_4.a.hint()

#step_4.a.solution_plot()
#step_4.b.hint()
# Check your answer (Run this code cell to receive credit!)

step_4.b.solution()