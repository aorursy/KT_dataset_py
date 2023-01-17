# Import required Libraries

import numpy as np 

import pandas as pd
# Input data files are available in the "../input/" directory.

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Read the data and store it in a dataframe

df=pd.read_csv('../input/cereal.csv')

df.head()
# Quick inspect of numerical data features

df.describe().round(3)