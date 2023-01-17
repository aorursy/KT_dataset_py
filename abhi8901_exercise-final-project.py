import pandas as pd

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

my_filepath = "../input/pokemon/pokemon_alopez247.csv"



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath)

# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.describe()
# Create a plot

plt.figure(figsize=(14,7))

sns.barplot(x='Attack', y='Defense', data=my_data) # Your code here



# Check that a figure appears below

step_4.check()
plt.figure(figsize=(15,7))

sns.scatterplot(x="Weight_kg", y="Height_m", hue="hasGender", data=my_data)
plt.figure(figsize=(10,6))

sns.swarmplot(x="Body_Style", y="Speed", data=my_data)
sns.lmplot(x="Generation", y="Sp_Atk", hue="hasMegaEvolution", data=my_data)
sns.lineplot(data=my_data[["Sp_Atk","HP","Pr_Male","Catch_Rate"]])