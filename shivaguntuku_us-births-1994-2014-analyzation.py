# import necessary python packages
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# load data files from US Births1994-2014 Directory
files = ['../input/US_births_1994-2003_CDC_NCHS.csv','../input/US_births_2000-2014_SSA.csv']
# concat the both files into single dataframe
df = pd.concat(map(pd.read_csv, files))
df.info

df.columns
df.tail()
