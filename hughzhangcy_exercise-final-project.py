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

my_filepath = "../input/my-test-data/Mobile Usage.csv"



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath)



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.head()
my_data.columns
sns.scatterplot(x=my_data['Peak_mins'], y=my_data['Total_mins'])
sns.scatterplot(x=my_data['OffPeak_mins'], y=my_data['Total_mins'])
sns.scatterplot(x=my_data['Weekend_mins'], y=my_data['Total_mins'])
sns.scatterplot(x=my_data['International_mins'], y=my_data['Total_mins'])
sns.kdeplot(data=my_data['Peak_mins'], shade=True)
sns.kdeplot(data=my_data['OffPeak_mins'], shade=True)
sns.kdeplot(data=my_data['Weekend_mins'], shade=True)
sns.kdeplot(data=my_data['International_mins'], shade=True)
# Create a plot

sns.jointplot(x=my_data['Peak_mins'], y=my_data['OffPeak_mins'], kind="kde")



# Check that a figure appears below

step_4.check()