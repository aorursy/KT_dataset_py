import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Path of the file to read

ign_filepath = "../input/ign_scores.csv"



# Fill in the line below to read the file into a variable ign_data

ign_data = pd.read_csv(ign_filepath, index_col="Platform")

# Print the data

ign_data # Your code here

ign_data.info()
#Easiet way to explore each column of the dataser is using describe command

ign_data.describe()
# Bar chart showing average score for racing games by platform

# Set the width and height of the figure

plt.figure(figsize=(8, 6))

# Bar chart showing average score for racing games by platform

sns.barplot(x=ign_data['Racing'], y=ign_data.index)

# Add label for horizontal axis

plt.xlabel("")

# Add label for vertical axis

plt.title("Average Score for Racing Games, by Platform")

# Heatmap showing average game score by platform and genre

# Set the width and height of the figure

plt.figure(figsize=(10,10))

# Heatmap showing average game score by platform and genre

sns.heatmap(ign_data, annot=True)

# Add label for horizontal axis

plt.xlabel("Genre")

# Add label for vertical axis

plt.title("Average Game Score, by Platform and Genre")