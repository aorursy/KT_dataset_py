# 5 Day Data Challenge, Day 1:

# Read in and summarize a .csv file

import pandas as pd



file = '../input/scrubbed.csv'

data = pd.read_csv(file)

data.describe()
