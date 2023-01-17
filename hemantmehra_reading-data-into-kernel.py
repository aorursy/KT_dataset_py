import numpy as np

import pandas as pd 



df = pd.read_csv('../input/cereal.csv') # load data from csv

df.head() # Gives first 5 rows
df.shape # (No. of rows, No. of cols)
df.describe() # Summary of numeric data