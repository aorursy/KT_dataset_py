# Import the necessary libraries

import pandas as pd

import numpy as np



# Read the data

dataset = pd.read_csv("../input/HospInfo.csv")
#Display the maximum columns to understand each column meaning

pd.options.display.max_columns = len(dataset.columns)



#Display the first 5 rows of data

dataset.head(1)
import numpy as np

dataset.groupby(['Hospital Type','Hospital Ownership'])['Provider ID'].count().idxmax()
#List the columns which are being compared with national average

cols = ['Mortality national comparison', 'Safety of care national comparison', 'Readmission national comparison',

       'Patient experience national comparison', 'Effectiveness of care national comparison', 'Timeliness of care national comparison',

       'Efficient use of medical imaging national comparison']



#Confirm the no of unique values in each column

for col in cols:

    print(dataset[col].unique())
#Let's convert the non numeric categorical values to numeric categorical values

averagedict = {

    'Same as the National average': 0,

    'Below the National average': -1,

    'Above the National average': 1,

    'Not Available': 2

}

cols = ['Mortality national comparison', 'Safety of care national comparison', 'Readmission national comparison',

       'Patient experience national comparison', 'Effectiveness of care national comparison', 'Timeliness of care national comparison',

       'Efficient use of medical imaging national comparison']
#Let's create the copy of the dataset since we are going to amend the dataset



dataset_copy = dataset.copy()
#Replace the non numerical categorical values with numeric cat

dataset_copy[cols] = dataset_copy[cols].replace(averagedict)
dataset_copy.head()
cols = ['Mortality national comparison', 'Safety of care national comparison', 'Readmission national comparison',

       'Patient experience national comparison', 'Effectiveness of care national comparison', 'Timeliness of care national comparison',

       'Efficient use of medical imaging national comparison']

for col in cols:

    dataset_copy = dataset_copy[dataset_copy[col] == 1]
dataset_copy
dataset_copy = dataset.copy()

#Replace the non numerical categorical values with numeric cat

dataset_copy[cols] = dataset_copy[cols].replace(averagedict)
colsnew = ['Mortality national comparison', 'Safety of care national comparison', 'Readmission national comparison','Effectiveness of care national comparison']

for col in colsnew:

    dataset_copy = dataset_copy[dataset_copy[col] == 1]
dataset_copy.groupby(['Hospital Type','Hospital Ownership'])['Provider ID'].count().idxmax()