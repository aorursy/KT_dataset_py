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

my_filepath = "../input/suicide-rates-overview-1985-to-2016/master.csv"



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

import pandas as pd

my_data = pd.read_csv(my_filepath, index_col='country', parse_dates=True)



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.head()


#country_count= my_data.groupby('country').count()

#print(country_count)

# Create a plot

plt.figure(figsize=(3,8))

sns.scatterplot(x="sex", y="suicides_no", data = my_data)



# Check that a figure appears below

step_4.check()
plt.figure(figsize=(22,19))



sns.catplot(x='age', y='suicides_no',jitter=False, data=my_data, height= 8, aspect= 1)

plt.figure(figsize=(20,15))

sns.catplot(x="age", y="suicides_no", hue="sex", kind="bar", data=my_data, height= 8, aspect= 1)