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
my_filepath = '../input/police-violence-in-the-us/deaths_and_stats.csv'

# Check for a valid filepath to a CSV file in a dataset
step_2.check()
# Fill in the line below: Read the file into a variable my_data
my_data = pd.read_csv(my_filepath)

# Check that a dataset has been uploaded into my_data
step_3.check()
# Print the first five rows of the data
my_data.head()
# Create a plot
state_violence = my_data.groupby('State').agg('sum') # Your code here

plt.figure(figsize=(20,10))
plt.xticks(rotation=30, horizontalalignment='right')
sns.barplot(x=state_violence.index, y=state_violence['Violent crimes 2013 (if reported by agency)'])
# Check that a figure appears below
step_4.check()
import numpy as np

vspy = state_violence[['Violent crimes 2013 (if reported by agency)',
                      'Violent crimes 2014 (if reported by agency)',
                      'Violent crimes 2015 (if reported by agency)',
                      'Violent crimes 2016 (if reported by agency)',
                      'Violent crimes 2017 (if reported by agency)',
                      'Violent crimes 2018 (if reported by agency)']]

x = np.arange(len(vspy))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots(figsize=(25,10))
i = 0
for elt in vspy.columns:
    barplot = ax.bar(x + width/2 + (i-3)*width, vspy[elt], width, label=elt.split()[2])
    i+=1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of violent crimes')
ax.set_title('Violent crimes per state and year')
ax.set_xticks(x)
ax.set_xticklabels(vspy.index, rotation=30, horizontalalignment='right')
ax.legend()

fig.tight_layout()

plt.show()