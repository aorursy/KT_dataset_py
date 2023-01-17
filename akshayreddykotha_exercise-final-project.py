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

my_filepath = "../input/team-performance-before-cricket-world-cup-2019/icc_mens_ranking.csv"



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath, encoding = "ISO-8859-1", index_col="Position")



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data
#my_data['Points']= str(my_data['Points'])

for each in my_data:

    each[2].replace(",","")
my_data.head()
# Create a plot

 # Your code here

sns.barplot(x=my_data["Rating"], y=my_data["Team"])



plt.title("Team ratings before the CWC 2019")



# Check that a figure appears below

step_4.check()
versus_data = pd.read_csv("../input/team-performance-before-cricket-world-cup-2019/recent_team_history-versus_opponent.csv")
versus_data.head()
versus_data.dtypes
# Creating the matrix to generate heatmap

vd_pivot = versus_data.pivot(index='Team', columns='Opponent',values='Ave')
plt.figure(figsize=(12,8))



sns.heatmap(data = vd_pivot, annot=True, fmt='g')



plt.title("Average runs scored by each batsman of any team against their opponent 2015-2019")