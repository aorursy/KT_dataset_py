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
my_filepath = "../input/dataisbeautiful/r_dataisbeautiful_posts.csv"

# Check for a valid filepath to a CSV file in a dataset
step_2.check()
# Fill in the line below: Read the file into a variable my_data
my_data = pd.read_csv(my_filepath, low_memory=False)


# Check that a dataset has been uploaded into my_data
step_3.check()
# Print the first five rows of the data
my_data.head()
# Create a plot
my_data.sort_values(by='total_awards_received', ascending=False, inplace=True)
my_clean_data = my_data[0:11][['author', 'total_awards_received']]

# Set the width and height of the figure
plt.figure(figsize=(10,6))

# Add title
plt.title("Awards per Author")

# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=my_clean_data['author'], y=my_clean_data['total_awards_received'])

# Add label for vertical axis
plt.ylabel("N of Awards")
plt.xlabel("Author")
plt.xticks(rotation="90")

# Check that a figure appears below
step_4.check()
plt.show()