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

my_filepath = '../input/canadian-car-accidents-19942014/NCDB_1999_to_2014.csv'



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# This jupyter notebook takes ideas and information from: https://www.kaggle.com/lastdruid/collision-descriptive-analysis-and-visualization 

# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath)



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.head()




print(my_data.columns )



# Check non numeric values

my_data[my_data['C_MNTH'].str.contains('[^0-9]')|

         my_data['C_WDAY'].str.contains('[^0-9]')|

         my_data['C_HOUR'].str.contains('[^0-9]')]



my_data.head()



#Remove all special values (unknown to us) in date-time columns, prepare for using date series as index.

#Make a copy "df" for further analysis, avoid mess up the original data "accident".



import numpy as np

df = my_data[:]

df[['C_MNTH','C_WDAY','C_HOUR']] = df[['C_MNTH','C_WDAY','C_HOUR']].replace('[^0-9]+',np.nan,regex=True)

df.dropna(axis=0,subset=['C_MNTH','C_WDAY','C_HOUR'],inplace=True)



df.head()



# Create a plot

X = df['C_HOUR'] # Your code here

y = df['C_CONF']



sns.barplot(X, y)



# Check that a figure appears below

step_4.check()