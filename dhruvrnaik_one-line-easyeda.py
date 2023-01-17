import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling



print ("pandas version: ",pd.__version__)

import os

print("Files = ", os.listdir("../input"))



# Load data

df = pd.read_csv('../input/zomato.csv')



report = pandas_profiling.ProfileReport(df)

report