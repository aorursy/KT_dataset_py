# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime as dt

import math

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
app_df = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')

app_df.head()
# Check for NaN values in the 'Last Updated Column'

app_df['Last Updated'].isnull().sum()
# The 'Last Updated' column has a perfect format of the time : "Month Date, Year"



'''

This is a function to check if the format is same with althrough the column. If not, drop the row with urregular format

'''

def checkformat(key,val):

    try:

        # strptime("June 4, 2018","%B %d, %Y") checks for the format and converts to datetime object

        return (dt.now()- dt.strptime(val,"%B %d, %Y")).days

    except:

        print("Found a bad format at index:{}. Found value:{}".format(key,val))

        app_df.drop(index=key,inplace=True)         # Drop the row with bad format

        app_df.reset_index(drop=True,inplace=True)



# Loop over the 'Last Updated' column to check for common format



for key,val in enumerate(app_df['Last Updated']):

    checkformat(key,val)

'''

Function to convert the values in the 'Last Updated' column into datetime object 

and compare with the current date to get the difference in time in DAYS 

'''

my_list=[]

def compare_time(key,val):

    try:

        return (dt.now()- dt.strptime(val,"%B %d, %Y")).days

    except:

        print("Value Error")



for key,val in enumerate(app_df['Last Updated']):

    my_list.append(compare_time(key,val))



# Append a column called 'Diff_Time' that gives the data about how many days it has been since last update

app_df['Updated X days ago'] = my_list

# Quick check on the newly appended column 'Updated X days ago'

app_df.head()
# Sort the values in the "Updated X days ago" column in the descending order and get the first row



app_df.sort_values(by='Updated X days ago',axis=0,ascending=False,inplace=True)

app_df.head() # After Sorting
print("App that was updated long back",app_df.iloc[0]['App'])

print("App that was recently updated:",app_df.iloc[-1]['App'])