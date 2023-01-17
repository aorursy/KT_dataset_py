from learntools.core import binder

binder.bind(globals())

from learntools.data_cleaning.ex4 import *

print("Setup Complete")
# modules we'll use

import pandas as pd

import numpy as np



# helpful character encoding module

import chardet



# set seed for reproducibility

np.random.seed(0)
sample_entry = b'\xa7A\xa6n'

print(sample_entry)

print('data type:', type(sample_entry))
new_entry = ____



# Check your answer

q1.check()
# Lines below will give you a hint or solution code

#q1.hint()

#q1.solution()
# TODO: Load in the DataFrame correctly.

police_killings = ____



# Check your answer

q2.check()
# (Optional) Use this code cell for any additional work.
# Lines below will give you a hint or solution code

#q2.hint()

#q2.solution()
# TODO: Save the police killings dataset to CSV

____



# Check your answer

q3.check()
# Lines below will give you a hint or solution code

#q3.hint()

#q3.solution()