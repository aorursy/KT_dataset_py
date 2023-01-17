import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

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
ign_data.describe()
ign_data.head()
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
# Print the data

ign_data # Your code here
ign_data.loc['PC']
ign_data.loc['PC'].max()
round(ign_data.loc['PC'].max(),6)
# Fill in the line below: What is the highest average score received by PC games,

# for any platform?

high_score = round(ign_data.loc['PC'].max(),6)



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

plt.figure(figsize=(25,6)) # Your code here

sns.barplot(y=ign_data['Racing'], x=ign_data.index)

# Check your answer

step_3.a.check()
# Lines below will give you a hint or solution code

#step_3.a.hint()

#step_3.a.solution_plot()
#step_3.b.hint()
#step_3.b.solution()
# Heatmap showing average game score by platform and genre

# Your code here

plt.figure(figsize=(10,10))

sns.heatmap(ign_data, annot=True)

# Check your answer

step_4.a.check()
# Lines below will give you a hint or solution code

#step_4.a.hint()

# step_4.a.solution_plot()
#step_4.b.hint()
step_4.b.solution()