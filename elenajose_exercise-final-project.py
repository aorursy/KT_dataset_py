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
# Check for a dataset with a CSV file
iris_flo = pd.read_csv("../input/iris-flowers/Iris_flowers.csv", index_col = "Id")
step_1.check()
# Fill in the line below: Specify the path of the CSV file to read
my_filepath = "../input/iris-flowers/Iris_flowers.csv"

# Check for a valid filepath to a CSV file in a dataset
step_2.check()
# Fill in the line below: Read the file into a variable my_data
my_data = pd.read_csv("../input/iris-flowers/Iris_flowers.csv", index_col = "Id")

# Check that a dataset has been uploaded into my_data
step_3.check()
# Print the first five rows of the data
my_data.head()
# Create a plot
sns.lineplot(data=my_data['PetalLengthCm']) # Your code here

# Check that a figure appears below
step_4.check()