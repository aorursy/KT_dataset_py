import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex7 import *

print("Setup Complete")




step_1.check()
# Fill in the line below: Specify the path of the CSV file to read

my_filepath = "../input/videogamesales/vgsales.csv"



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath, index_col="Year")



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.head()
# Create a plot

plt.figure(figsize=(30,20))

plt.title("Video Games with sales greater than 80,000 by Platform")



sns.scatterplot(x=my_data.index, y=my_data["Genre"])

plt.ylabel("Genre")

plt.legend()



# Check that a figure appears below

step_4.check()