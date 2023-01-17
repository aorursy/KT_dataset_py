# import libraries
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read in and show the first few lines
cereal = pd.read_csv("../input/cereal.csv")
cereal.head()
# look at dimension of the data for the number of rows and columns
print(cereal.shape)

# look at a summary of the numeric columns
cereal.describe()
# This version will show all the columns, including non-numeric
cereal.describe(include="all")