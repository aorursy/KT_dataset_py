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

my_filepath = "../input/traffic-collision-data-from-2010-to-present.csv"



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath)



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.head()
def turnNA(X,f):

    for i in range(len(X[f])):

        if str(X[f][i])=='nan':

            X.set_value(i, f, -1)

    return X

my_data = turnNA(my_data,'Victim Age')

my_data = turnNA(my_data,'Time Occurred')



# Line chart showing how FIFA rankings evolved over time 

# sns.lineplot(data=my_data['Victim Age'], label="Victim Age")



# Set the width and height of the figure

plt.figure(figsize=(16,6))

# sns.scatterplot(x=my_data['Reporting District'], y=my_data['Victim Age'])

# sns.regplot(x=my_data['Reporting District'], y=my_data['Victim Age'])

sns.scatterplot(x=my_data['Time Occurred'], y=my_data['Victim Age'], hue=my_data['Victim Sex'])

# sns.lmplot(x="Time Occurred", y="Victim Age", hue="Victim Sex", data=my_data)

# Add title

plt.title("Victim age by collision time")

# Add label for horizontal axis

# plt.xlabel("Collision time")

# plt.ylabel("Victim age")



# Set the width and height of the figure

plt.figure(figsize=(16,6))

# Histogram 

# sns.distplot(a=my_data['Victim Age'], kde=False)

# sns.kdeplot(data=my_data['Victim Age'], shade=True)

# 2D KDE plot

sns.jointplot(x=my_data['Time Occurred'], y=my_data['Victim Age'], kind="kde")



# Check that a figure appears below

step_4.check()