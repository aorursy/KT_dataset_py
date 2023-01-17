# Importing libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualization

import plotly.plotly as py

import plotly.graph_objs as go

import os

# Checking files in the directory

print(os.listdir("../input"))

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()
# importing and installing .csv validator

import sys

!{sys.executable} -m pip install csvvalidator
# Read the data, print out number of rows, column names, and data head

data = pd.read_csv("../input/crisis-data.csv")

print(data.columns)

print(len(data))

print(data.head())
data.head(20).loc[:,'Subject Veteran Indicator']
from csvvalidator import *



field_names = ('Template ID',

               'Precinct',

               'Reported Date',

               'Subject Veteran Indicator'

               )



validator = CSVValidator(field_names)



validator.add_value_check('Template ID', # the name of the field

                          int, # a function that accepts a single argument and 

                               # raises a `ValueError` if the value is not valid.

                               # Here checks that "key" is an integer.

                          'EX1', # code for exception

                          'ID must be an integer'# message to report if error thrown

                         )



validator.add_value_check('Reported Date', 

                          # check for a date with the sepcified format

                          datetime_string('%Y-%m-%d'), 

                          'EX2',

                          'invalid date'

                         )



validator.add_value_check('Subject Veteran Indicator', 

                          enumeration('Yes', 'No', 'Unknown', '-'),

                          'EX4', 

                          'Veteran indicator input not recognized')



validator.add_value_check('Precinct', 

                          enumeration('North', 'South', 'East', 'West', 'Southwest', 'Unknown'),

                          'EX5', 

                          'Precinct input not recognized')
# Looking at the complaints of the initial calls

complaint_type_count = data['Initial Call Type'].unique()

print(len(complaint_type_count))

print(complaint_type_count)

initial_call_counts = data.groupby(['Initial Call Type']).count().sort_values('Template ID', ascending=False)

print(initial_call_counts)
# Looking at complaint categories of the call from the final report

final_calls = data.groupby(['Final Call Type']).count().sort_values('Template ID',ascending=False)

print(final_calls.iloc[:,0:1])
y_pos = np.arange(len(final_calls))[:10]

vals = list(map(lambda x: x.item(),final_calls.iloc[:,0:1].values))[:10]

lbls = final_calls.index[:10]

plt.bar(y_pos, vals, align='center', alpha=0.5)

plt.xticks(y_pos,lbls,rotation=90)

plt.ylabel('Number of Calls')

plt.title('Top Ten Seattle 911 Calls')
data['Precinct'].value_counts().plot(kind='pie')
#precinct_count = data.groupby(['Precinct']).count().sort_values('Template ID', ascending=False).iloc[:,:1]

veterans = data[data['Subject Veteran Indicator'] == 'Yes']

veteran_count = veterans.groupby(['Reported Date']).count().sort_values('Reported Date', ascending=False)

#veteran_count = veteran_count.drop(veteran_count.index[1314])

print(veteran_count)
data2 = [go.Scatter(x=veteran_count.index,y=veteran_count.loc[:,'Subject Veteran Indicator'])]



layout = dict(title = "Number of Veteran Calls per Day",

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

fig = dict(data = data2, layout = layout)

iplot(fig)
