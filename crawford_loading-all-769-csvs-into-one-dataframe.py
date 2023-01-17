import numpy as np

import pandas as pd

import glob

import os
# See how many files there are in the directory. 

# "!" commands are called "magic commands" and allow you to use bash

file_dir = '../input/kospi'

!ls $file_dir
# Number of files we are dealing with

!ls $file_dir | wc -l
# Get a python list of csv files

files = glob.glob(os.path.join(file_dir, "*.csv"))
# Look at a few to see how we can merge them

df1 = pd.read_csv(files[0])

df2 = pd.read_csv(files[1])

df3 = pd.read_csv(files[2])



print(df1.head(), "\n")

print(df2.head(), "\n")

print(df3.head(), "\n")
# Make a list of dataframes while adding a stick_ticker column

dataframes = [pd.read_csv(file).assign(stock_ticker=os.path.basename(file).strip(".csv")) for file in files]

# Concatenate all the dataframes into one

df = pd.concat(dataframes, ignore_index=True)
df.head()
df.shape