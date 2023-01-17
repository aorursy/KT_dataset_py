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
# Path of the file to read
ign_filepath = "../input/ign_scores.csv"

# Fill in the line below to read the file into a variable ign_data
ign_data = pd.read_csv(ign_filepath, index_col='Platform')
# Print the data
ign_data
# Fill in the line below: What is the highest average score received by PC games,
# for any platform?
high_score = 7.759930

# Fill in the line below: On the Playstation Vita platform, which genre has the 
# lowest average score? Please provide the name of the column, and put your answer 
# in single quotes (e.g., 'Action', 'Adventure', 'Fighting', etc.)
worst_genre = 'Simulation'
# Bar chart showing average score for racing games by platform
plt.figure(figsize=(20, 15))
plt.title('Avg Score for Racing Games, by Platform')
sns.barplot(x=ign_data['Racing'], y=ign_data.index)
plt.xlabel('Score')
# I cannot expect a racing game for Wii to receive a high rating. A racing game for Xbox One looks like a better option.
# Heatmap showing average game score by platform and genre
plt.figure(figsize = (15, 15))
sns.heatmap(data = ign_data, annot = True)
plt.xlabel('Genre')
plt.title('Heatmap of ratings, by platform and genre')
# Highest rating: PS4, Simulation (9.2); Lowest rating: Game Boy Color, Fighting and Shooting (4.5).