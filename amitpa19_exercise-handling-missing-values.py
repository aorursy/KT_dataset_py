from learntools.core import binder

binder.bind(globals())

from learntools.data_cleaning.ex1 import *

print("Setup Complete")
# modules we'll use

import pandas as pd

import numpy as np



# read in all our data

sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")



# set seed for reproducibility

np.random.seed(0) 
# TODO: Your code here!

sf_permits.head(5)
# Check your answer (Run this code cell to receive credit!)

q1.check()
# Line below will give you a hint

#q1.hint()
# TODO: Your code here!

total_cells = np.product(sf_permits.shape)

missing_values_count = sf_permits.isnull().sum()

total_missing = missing_values_count.sum()

percent_missing = (total_missing/total_cells)*100

print(percent_missing)

# Check your answer

q2.check()
# Lines below will give you a hint or solution code

#q2.hint()

#q2.solution()
# Check your answer (Run this code cell to receive credit!)

sf_permits[["Street Number Suffix","Zipcode"]].head()

q3.check()
# Line below will give you a hint

#q3.hint()
# TODO: Your code here!

sf_permits_new = sf_permits.dropna()

sf_permits_new.shape[0]
# Check your answer (Run this code cell to receive credit!)

q4.check()
# Line below will give you a hint

#q4.hint()
# TODO: Your code here

sf_permits_with_na_dropped = sf_permits.dropna(axis=1)

cols_in_na_dropped = sf_permits_with_na_dropped.shape[1]

dropped_columns = (sf_permits.shape[1]-cols_in_na_dropped)



# Check your answer

q5.check()
# Lines below will give you a hint or solution code

#q5.hint()

#q5.solution()
# TODO: Your code here

sf_permits_with_na_imputed = sf_permits.fillna(method='bfill', axis=0).fillna(0)



# Check your answer

q6.check()
# Lines below will give you a hint or solution code

#q6.hint()

#q6.solution()