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

my_filepath = "../input/novel-corona-virus-2019-dataset/covid_19_data.csv"



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath, parse_dates=["Last Update"])

my_data.rename(columns={"ObservationDate":"Date", "Country/Region":"Country"}, inplace=True)

# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.head()
# Visualise relevant data for worldwide cases

my_data.drop(columns="SNo", inplace=True)

my_data.groupby("Date").sum().reset_index()
# Assign relevant data needed to variable total_cases

total_cases = my_data.groupby("Date").sum().reset_index()
# Visualise total_cases data

total_cases
#Create a plot

sns.set_style("darkgrid")

plt.figure(figsize=(15, 8))

plt.title("Cumulative Worldwide Cases of COVID-19 (YTD)", fontsize=25)

sns.lineplot(x="Date", y="Confirmed", data=total_cases, label="Confirmed", color="blue")

sns.lineplot(x="Date", y="Deaths", data=total_cases, label="Deaths", color="red")

sns.lineplot(x="Date", y="Recovered", data=total_cases, label="Recovered", color="green")

plt.xticks(rotation=90)

plt.xlabel("Date",fontsize=20)

plt.ylabel("Total no. of cases", fontsize=20)

plt.legend()



# Your code here



# Check that a figure appears below

step_4.check()