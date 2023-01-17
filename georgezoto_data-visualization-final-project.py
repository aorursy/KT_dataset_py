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
step_1.check()
# Fill in the line below: Specify the path of the CSV file to read
my_filepath = '../input/run-or-walk/dataset.csv'

# Check for a valid filepath to a CSV file in a dataset
step_2.check()
# Fill in the line below: Read the file into a variable my_data
my_data = pd.read_csv(my_filepath, index_col='date', parse_dates=True)

# Check that a dataset has been uploaded into my_data
step_3.check()
# Print the first five rows of the data
my_data.head()
my_data.describe()
my_data['username'].value_counts()
my_data.index
# Create a plot
my_figsize = (16, 9)
plt.figure(figsize=my_figsize)

plt.title('Accelaration linechart')
sns.lineplot(data=my_data['acceleration_x'], label='acceleration_x')
sns.lineplot(data=my_data['acceleration_y'], label='acceleration_y')
sns.lineplot(data=my_data['acceleration_z'], label='acceleration_z')
plt.xlabel("Date")

# Check that a figure appears below
step_4.check()
plt.figure(figsize=my_figsize)
plt.title('Accelaration distribution')
sns.distplot(a=my_data['acceleration_x'], label='acceleration_x', kde=False)
sns.distplot(a=my_data['acceleration_y'], label='acceleration_y', kde=False)
sns.distplot(a=my_data['acceleration_z'], label='acceleration_z', kde=False)
plt.legend()