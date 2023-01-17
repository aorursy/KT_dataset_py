'''

Created on - November 15, 2017

Hi everyone!

My name is Paul Lee. I am a software engineer and beginner data scientist. 

I am trying to learn data science while trying to solve a real life problem, 

so any tips would be greatly appreciated. Thank you so much!



This will be a constant work in progress when I have the time, let me know any changes I should make!

'''



import numpy as np 

import pandas as pd 

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
#Read in the file and put it into a dataframe

df = pd.read_csv("../input/BPD_Part_1_Victim_Based_Crime_Data.csv")
#Get a quick glispe of what's in there

df.head(5)
'''

Trying to figure out some statistics such as how many rows, col, size, just to get a better understanding

of what i'm dealing with.

'''

dataframe_rows, dataframe_cols = df.shape

print("The size of the DF is",df.size,".")

print("The dimensions of the DF is",dataframe_rows,"rows and",dataframe_cols,"columns.")
'''

I got a quick glimpse of the data and noticed that Total Incidents is probably the same across the whole

dataset. I will test that now.

'''

df['Total Incidents'].value_counts()
'''

Just as I thought, the total incidents column is all the same, so I'm just going to drop that

'''

df = df.drop('Total Incidents',axis=1)
'''

I want to convert all columns to lowercase so that's its easier to work with

'''

df.columns = [col.lower() for col in df.columns]
'''

What types of crime is happening? What's the most common over the last 5-6 years

'''

df['description'].value_counts()