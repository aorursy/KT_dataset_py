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

my_filepath = "../input/data-for-datavis/spotify.csv"



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

import pandas as pd

my_data = melbourne_data = pd.read_csv(my_filepath) 



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.head()
# Create a plot

import matplotlib.pyplot as plt

import seaborn as sns

____ # Your code here

#plt.plot(my_data.Despacito, my_data.Date, label='linear')

plt.title("my visualization ")

sns.scatterplot(x=my_data.Despacito, y=my_data.Date)



# Check that a figure appears below

step_4.check()