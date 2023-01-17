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

my_filepath = '../input/uncover/UNCOVER/harvard_global_health_institute/hospital-capacity-by-state-40-population-contracted.csv'



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath,index_col = "state",parse_dates=True)



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.head()


# Create a plot

sns.barplot(data=my_data)



# Check that a figure appears below

step_4.check()
plt.figure(figsize=(20,10))

plt.title("Number of Available hospital beds in various states")

sns.barplot(x=my_data.index,y=my_data['available_hospital_beds'])

plt.ylabel("Available Hospital Beds")
# Set the width and height of the figure

plt.figure(figsize=(40,20))



# Add title

plt.title("Number of hospital beds in various states")



# Heatmap showing average arrival delay for each airline by month

sns.heatmap(my_data, annot=True)



# Add label for horizontal axis

plt.xlabel("state")
# Histogram 

sns.distplot(a=my_data['icu_beds_needed_eighteen_months'], kde=False)
sns.kdeplot(data=my_data['icu_bed_occupancy_rate'], shade=True)
# 2D KDE plot

sns.jointplot(x=my_data['potentially_available_hospital_beds'], y=my_data['percentage_of_total_icu_beds_needed_six_months'], kind="kde")
sns.lineplot(x=my_data['percentage_of_total_icu_beds_needed_six_months'],y='population_65',data=my_data)