# trying to understand what can be done with this data

# use pandas profiling
import pandas as pd

import numpy as np 

import os
df = pd.read_excel('../input/national-anxiety-survey/depression servey.xlsx')

df.head(1)

                   
df.info()
from pandas_profiling import ProfileReport

ProfileReport(df)