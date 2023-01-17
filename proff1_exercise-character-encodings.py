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
new_entry = sample_entry

new_entry = new_entry.decode('utf-8',errors='replace')

new_entry

# Check your answer

q1.check()
# TODO: Load in the DataFrame correctly.

police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv")



# Check your answer

q2.check()

police_killings.head()
# look at the first ten thousand bytes to guess the character encoding

with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:

    result = chardet.detect(rawdata.read(30000))



# check what the character encoding might be

print(result)

q2.check()
# read in the file with the encoding detected by chardet

police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='Windows-1252')

police_killings.head()

q2.check()
# TODO: Save the police killings dataset to CSV

police_killings.to_csv("police_killings_utf8.csv")



q3.check()