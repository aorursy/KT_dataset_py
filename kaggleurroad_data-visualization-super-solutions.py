import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

# Set up code checking

import os

if not os.path.exists("../input/ign_scores.csv"):

    os.symlink("../input/data-for-datavis/ign_scores.csv", "../input/ign_scores.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex3 import *

# Path of the file to read

ign_filepath = "../input/ign_scores.csv"



# Fill in the line below to read the file into a variable ign_data

ign_data =pd.read_csv(ign_filepath,index_col="Platform")

# Print the data

print(ign_data) # Your code here
# Fill in the line below: What is the highest average score received by PC games,

# for any platform?



high_score = 7.759930



# Fill in the line below: On the Playstation Vita platform, which genre has the 

# lowest average score? Please provide the name of the column, and put your answer 

# in single quotes (e.g., 'Action', 'Adventure', 'Fighting', etc.)

worst_genre = 'Simulation'





# Bar chart showing average score for racing games by platform

plt.figure(figsize=(10, 6))

# Bar chart showing average score for racing games by platform

sns.barplot(x=ign_data['Racing'], y=ign_data.index)

# Add label for horizontal axis

plt.xlabel("average score")

# Add label for vertical axis

plt.title("Average Score for Racing Games, by Platform")# Your code here



# Check your answer

step_3.a.check()
#  showing average game score by platform and genre



plt.figure(figsize=(10,10))

# Heatmap showing average game score by platform and genre

sns.heatmap(ign_data, annot=True)

# Add label for horizontal axis

plt.xlabel("Genre")

# Add label for vertical axis

plt.title("Average Game Score, by Platform and Genre")

# Check your answer

step_4.a.check()