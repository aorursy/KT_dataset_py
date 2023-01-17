import pandas as pd
import numpy as np
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
my_filepath = "../input/fivethirtyeight-comic-characters-dataset/dc-wikia-data.csv"

# Check for a valid filepath to a CSV file in a dataset
step_2.check()
# Fill in the line below: Read the file into a variable my_data
my_data = pd.read_csv(my_filepath)

# Check that a dataset has been uploaded into my_data
step_3.check()
# Print the first five rows of the data
my_data.head()
# Drop unnecessary column
my_data = my_data.drop(['page_id','urlslug'], axis=1)
my_data
straight = my_data.drop(my_data[ my_data['GSM'] == 'Bisexual Characters'].index).drop(my_data[ my_data['GSM'] == 'Homosexual Characters'].index).drop(my_data[my_data['SEX'] == 'Transgender Characters'].index)
straight.sort_values(by=['APPEARANCES'],ascending=False).head()
no_straight = my_data[ my_data['GSM'] == 'Bisexual Characters']
no_straight = no_straight.append(my_data[my_data['GSM'] == 'Homosexual Characters'])
no_straight = no_straight.append(my_data[ my_data['SEX'] == 'Transgender Characters'])
no_straight.sort_values(by=['APPEARANCES'],ascending=False).head()
# Create a plot
# Your code here

# Histograms for each species
sns.distplot(a=straight['YEAR'], label="Straight", kde=False)
sns.distplot(a=no_straight['YEAR'], label="No Straight", kde=False)

# Add title
plt.title("Distribution of DC Heroes by diversity")

# Force legend to appear
plt.legend()
plt.xlabel('Year of Appearence')
plt.ylabel('N° of Heroes')
step_4.check()