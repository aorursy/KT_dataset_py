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

my_filepath = "../input/zomato-bangalore-restaurants/zomato.csv"



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath)



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.head()
# Create a plo

# data=my_data[['rate','votes']]

sns.scatterplot(x=my_data.index, y=my_data['votes'])# Your code here



# Check that a figure appears below

#step_4.check()
my_data.describe()
my_data.columns
sns.kdeplot( data=my_data['votes'])
plt.figure(figsize=(20,4))

sns.barplot(x=my_data['votes'].head(30), y=my_data.head(30).index )# Your code here