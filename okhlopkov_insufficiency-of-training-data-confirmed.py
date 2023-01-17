# Reading the data
import pandas as pd 
train = pd.read_csv("../input/train.csv", index_col='ID')

# Remove target to simplify the calculation of duplicates
target = train.pop("target")

# Find duplicate rows
t = train.duplicated(keep=False)
duplicated_indexes = t[t].index.values
print("Indexes of duplicated rows: {}".format(duplicated_indexes))

# Show target values for selected indexes
target.loc[duplicated_indexes]
import numpy as np
first_ind, second_ind = duplicated_indexes[0], duplicated_indexes[1]
new_target_val = np.exp((np.log(target.loc[first_ind]) + np.log(target.loc[second_ind])) / 2)

target = target.drop(first_ind)
target[second_ind] = new_target_val
train = train.drop(first_ind)