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

my_filepath = '../input/fivethirtyeight-comic-characters-dataset/dc-wikia-data.csv'



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath)



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.head()
# Create a plot

# ____ # Your code here



# Check that a figure appears below

plt.figure(figsize=(14,7))

sns.countplot(x=my_data['SEX'])



step_4.check()
plt.figure(figsize=(14,7))

sns.swarmplot(x=my_data['SEX'], y=my_data['APPEARANCES'])
plt.figure(figsize=(14,7))

sns.swarmplot(x=my_data['ALIGN'], y=my_data['APPEARANCES'])
plt.figure(figsize=(14,7))

sns.scatterplot(x=my_data['YEAR'], y=my_data['APPEARANCES'], hue=my_data['ALIGN'])