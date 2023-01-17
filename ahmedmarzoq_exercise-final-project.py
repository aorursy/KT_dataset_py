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

my_filepath = "../input/nyc-jobs.csv"



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath, index_col="Posting Date", parse_dates=True)



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.head()
# Create a plot

# Your code here

plt.figure(figsize=(18,6))

sns.lineplot(data=my_data['Salary Range From']) # Your code here

# Check that a figure appears below

step_4.check()
plt.figure(figsize=(14,20))

sns.countplot(y=my_data['Business Title'],palette="Set3")
plt.figure(figsize=(10,10))

sns.countplot(x=my_data['Posting Type'])
plt.figure(figsize=(15,25))

sns.countplot(y=my_data['Job Category'])
plt.figure(figsize=(14,7))

sns.swarmplot(y=my_data['# Of Positions'], x=my_data['Level'])
plt.figure(figsize=(14,7))

sns.countplot(x=my_data['Level'])