# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex2 import *
print("Setup Complete")
import pandas as pd
# FilePath to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'
# Iowa file path stored in Pandas DataFrame called home_data
home_data = pd.read_csv(iowa_file_path)
step_1.check()
# Lines below will give you a hint or solution code
#step_1.hint()
#step_1.solution()
# Print summary statistics in next line
import numpy as np
import pandas as pd
# summary of the home_data
home_data.describe()
# What is the average lot size (rounded to nearest integer)?
avg_lot_size = 10517



# As of today, how old is the newest home (current year - the date in which it was built)
newest_home_age = 10

# Checks your answers
step_2.check()
#step_2.hint()
#step_2.solution()