import pandas as pd

import numpy as np

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

my_filepath = "../input/sales-plus-walmart/output.csv"



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath)



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.head()
my_data.shape
# Create a plot

plt.figure(figsize=(14,6))



plt.title("Walmart Sales")



sns.lineplot(x='Id', y='Weekly_Sales', data=my_data.head(100)) # Your code here



# Check that a figure appears below

step_4.check()