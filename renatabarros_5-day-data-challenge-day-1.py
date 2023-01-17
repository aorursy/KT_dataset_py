# Start by importing the basics



import numpy as np

import pandas as pd



# Read the csv file through pandas and summarise the data

# Obs. The 'include = 'all'' guarantees all columns (numeric or not) are summarised



ufo_df = pd.read_csv("../input/scrubbed.csv")

pd.DataFrame.describe(ufo_df, include = 'all')






