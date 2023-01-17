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
sf_permits.head()
# Check your answer (Run this code cell to receive credit!)
q1.check()
# Line below will give you a hint
#q1.hint()
# TODO: Your code here!
# total available entries in the dataframe
total_entry = np.product(sf_permits.shape)
total_entry

# total missing entries with NaN
entry_missing = sf_permits.isnull().sum()
total_missing_entry = total_missing.sum()

# percentage missing entries
percent_missing = (total_missing_entry/total_entry) * 100
percent_missing

# Check your answer
q2.check()
# Lines below will give you a hint or solution code
#q2.hint()
#q2.solution()
# Check your answer (Run this code cell to receive credit!)
q3.check()
# Line below will give you a hint
q3.hint()
# TODO: Your code here!
print('Number of rows in the dataset :', sf_permits.shape[0])
print('Number of rows without missing values :', sf_permits.dropna())
# Check your answer (Run this code cell to receive credit!)
q4.check()
# Line below will give you a hint
q4.hint()
# TODO: Your code here
sf_permits_with_na_dropped = sf_permits.dropna(axis=1)


total_no_columns = sf_permits.shape[1]

dropped_columns = total_no_columns - sf_permits_with_na_dropped.shape[1]
dropped_columns

# Check your answer
q5.check()
# Lines below will give you a hint or solution code
q5.hint()
q5.solution()
# TODO: Your code here
sf_permits_with_na_imputed = sf_permits.fillna(method='bfill', axis=0)
sf_permits_with_na_imputed  = sf_permits_with_na_imputed.fillna(0)
sf_permits_with_na_imputed.isnull().sum()
# Check your answer
q6.check()
# Lines below will give you a hint or solution code
#q6.hint()
#q6.solution()