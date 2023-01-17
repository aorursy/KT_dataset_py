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

# my_filepath = "../input/mount-rainier-weather-and-climbing-data/climbing_statistics.csv"

my_filepath = "../input/mount-rainier-weather-and-climbing-data/Rainier_Weather.csv"



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath, index_col="Date", parse_dates=True)



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.head()
print(list(my_data.columns))
# Create a plot

# sns.lmplot(x="Temperature AVG", y="Solare Radiation AVG", data=my_data)

label=my_data.columns[1]

label_2=my_data.columns[5]

sns.kdeplot(data=my_data[label], label="{}".format(label), shade=True)

sns.kdeplot(data=my_data[label_2], label="{}".format(label_2), shade=True)





# Check that a figure appears below

step_4.check()
label_joint=my_data.columns[1]

label_joint_2=my_data.columns[2]

sns.jointplot(x=my_data[label_joint], y=my_data[label_joint_2], kind="kde")
plt.figure(figsize=(9,4))



label = my_data.columns[1]

sns.lineplot(data=my_data[label], label="{}".format(label))
# plt.figure(figsize=(12,6))

# sns.heatmap(data=my_data, annot=False)

# sns.barplot(x=my_data.index, y=my_data['Temperature AVG'])