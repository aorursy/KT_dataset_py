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

my_filepath_est_num ='../input/malaria-dataset/estimated_numbers.csv'

my_filepath_inc_1000 = '../input/malaria-dataset/incidence_per_1000_pop_at_risk.csv'

my_filepath_rep_num = '../input/malaria-dataset/reported_numbers.csv'



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath_est_num)

my_data_est_num = pd.read_csv(my_filepath_est_num)

my_data_inc_1000 = pd.read_csv(my_filepath_inc_1000)

my_data_rep_num = pd.read_csv(my_filepath_rep_num)





# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data_est_num.head()
my_data_inc_1000.head()
my_data_rep_num.head()
plt.figure(figsize=(14,7))

sns.scatterplot(x='WHO Region', y='No. of cases', data=my_data_inc_1000)
plt.figure(figsize=(14,7))

sns.distplot(my_data_inc_1000['No. of cases'], kde=False)
# Create a plot

____ # Your code here

plt.figure(figsize=(14,7))

sns.barplot(x='WHO Region', y='No. of cases', data=my_data_inc_1000)



# Check that a figure appears below

step_4.check()
my_data_rep_num.columns
plt.figure(figsize=(14,7))



sns.scatterplot(x='No. of cases', y='No. of deaths', hue='WHO Region', data=my_data_rep_num)
plt.figure(figsize=(14,7))



sns.barplot(x='Year', y='No. of cases', data=my_data_rep_num)
my_data_est_num.columns
plt.figure(figsize=(14,7))

# my_data_rep_num.head()

sns.scatterplot(x='WHO Region', y='No. of cases', data=my_data_inc_1000, hue='WHO Region')