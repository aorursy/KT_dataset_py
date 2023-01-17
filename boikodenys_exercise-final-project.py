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

my_filepath = '../input/fifa-world-cup/WorldCups.csv'



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath, index_col = 0)



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.head()
# Create a plot



plt.figure(figsize = (16, 6))

plt.title("Lineplot")

sns.lineplot(data = my_data.loc[:, ['GoalsScored', 'QualifiedTeams', 'MatchesPlayed']]) # Your code here #lineplot





# Check that a figure appears below

step_4.check()
plt.figure(figsize = (16, 6))



# Add title

plt.title('Barplot')



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(x = my_data.index, y = my_data['GoalsScored'])



# Add label for vertical axis

plt.ylabel('Goals scored in FIFA WC')