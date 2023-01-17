import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex7 import *

print("Setup Complete")
# Check for a dataset with a CSV file

step_1.check()
import os

print(os.listdir("../input"))
# specify the path of the CSV file to read

my_filepath = "../input/fivethirtyeight-comic-characters-dataset/marvel-wikia-data.csv/marvel-wikia-data.csv"



# check for a valid filepath to a CSV file in a dataset

step_2.check()
# read the file into a variable my_data

my_data = pd.read_csv(my_filepath, index_col="Year", parse_dates=True)



# check that a dataset has been uploaded into my_data

step_3.check()
# print the first five rows of the data

my_data.head()
# Create a plot

sns.lineplot(data=my_data["APPEARANCES"])



# Check that a figure appears below

step_4.check()