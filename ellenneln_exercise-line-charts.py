import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex2 import *

print("Setup Complete")
# Path of the file to read

museum_filepath = "../input/museum_visitors.csv"



# Fill in the line below to read the file into a variable museum_data

museum_data = pd.read_csv(museum_filepath, index_col='Date', parse_dates=True)



# Run the line below with no changes to check that you've loaded the data correctly

step_1.check()
# Uncomment the line below to receive a hint

#step_1.hint()

# Uncomment the line below to see the solution

#step_1.solution()
# Print the last five rows of the data 

museum_data.tail() # Your code here
# Fill in the line below: How many visitors did the Chinese American Museum 

# receive in July 2018?

ca_museum_jul18 = museum_data.iloc[:,2]['2018-07-01']

#ca_museum_jul18



# Fill in the line below: In October 2018, how many more visitors did Avila 

# Adobe receive than the Firehouse Museum?

avila_oct18 = museum_data.iloc[:,0]['2018-10-01'] - museum_data.iloc[:,1]['2018-10-01']

#avila_oct18



# Check your answers

step_2.check()
# Lines below will give you a hint or solution code

#step_2.hint()

step_2.solution()
# Line chart showing the number of visitors to each museum over time

ax = sns.lineplot(data=museum_data) # Your code here

ax.set(xlabel='Period', ylabel='Nr of Visitors')



# Check your answer

step_3.check()

plt.show()
# Lines below will give you a hint or solution code

#step_3.hint()

#step_3.solution_plot()
# Line plot showing the number of visitors to Avila Adobe over time

# Note since his objective is to study the seasonality the figure is created with a higher lenght than width

plt.figure(figsize=(12,6))

# Add title

plt.title("Number of monthly visitors at Avila Adobe")

sns.lineplot(data=museum_data.iloc[:,0]) # Your code here

plt.xlabel('Date')



# Check your answer

step_4.a.check()

plt.show()
# Lines below will give you a hint or solution code

#step_4.a.hint()

step_4.a.solution_plot()
#step_4.b.hint()

import calendar

import numpy as np

avilaVisitors = museum_data.iloc[:,0]

overall_avg = avilaVisitors.mean()

months = range(1, 13)

MM = []

YY = []

MMM = []

MMMM = []

for i in range(len(avilaVisitors)):

    MM.append(((avilaVisitors.index[i]).to_pydatetime()).month)

    YY.append(((avilaVisitors.index[i]).to_pydatetime()).year)    

MMMM = [calendar.month_name[i] for i in MM]

MMM = [x[:3] for x in MMMM]



cats = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']



avilaVisitorsDF = pd.DataFrame()

avilaVisitorsDF['Visitors'] = avilaVisitors

avilaVisitorsDF['MM'] = MM

avilaVisitorsDF['MMMM'] = MMMM

avilaVisitorsDF['MMM'] = MMM

avilaVisitorsDF['MMM'] = pd.Categorical(avilaVisitorsDF['MMM'], ordered=True, categories=cats)

avilaVisitorsDF['YY'] = YY

avilaVisitorsDF.index = avilaVisitors.index



seasonality_idx = (avilaVisitorsDF.groupby('MM')).Visitors.mean() / overall_avg

print(seasonality_idx[1])



fig, ax = plt.subplots(figsize=(12,6))

plt.title("Monthly Visitors to Avila Adobe with Seasonality Index")



sns.lineplot(x = 'MMM', y = 'Visitors', data=avilaVisitorsDF[avilaVisitorsDF['YY'] == 2014], label='2014')

sns.lineplot(x = 'MMM', y = 'Visitors', data=avilaVisitorsDF[avilaVisitorsDF['YY'] == 2015], label='2015')

sns.lineplot(x = 'MMM', y = 'Visitors', data=avilaVisitorsDF[avilaVisitorsDF['YY'] == 2016], label='2016')

sns.lineplot(x = 'MMM', y = 'Visitors', data=avilaVisitorsDF[avilaVisitorsDF['YY'] == 2017], label='2017')

sns.lineplot(x = 'MMM', y = 'Visitors', data=avilaVisitorsDF[avilaVisitorsDF['YY'] == 2018], label='2018')



style = dict(size=10, color='gray')

for month in cats:

    ax.text(month, 43000, str(seasonality_idx[cats.index(month)+1])[:4], ha='center', **style)



ax.set(ylim=(0, 45000))



plt.xlabel("Month")

plt.ylabel("Visitors")

plt.legend(loc='lower right')



plt.show()
step_4.b.solution()