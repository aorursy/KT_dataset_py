import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import matplotlib.pyplot as plt

import os
# List the file sint eh directory

file_dir = '../input/'

!ls -1 $file_dir
# Create list of files

files = glob.glob(os.path.join(file_dir, "*.csv"))
# Get header from one of teh CSVs

header = pd.read_csv(files[0], nrows=1)



# Make a list of dataframes and use the file name to create a "date" column

dataframes = [pd.read_csv(file, header=0).assign(date=os.path.basename(file).strip(".csv")) for file in files]



# Concatenate all the dataframes into one

df = pd.concat(dataframes, ignore_index=True)



# Remove characters from column names

df.rename(columns = {"% 1h": "1h", "% 2h": "2h", "% 7d": "7d"}, inplace=True)



# Remove special characters from values

df.replace(['\%', '\$', '\?', '\,', '\*'], ['','', np.nan, '', ''], regex=True, inplace=True)



# Change some numerical columns to actual numeric characters

cols = ["Cap de mercat", "Preu", "Oferta circulant", "Volume 24 hores", "1h", "2h", "7d"]

for col in df.columns:

    if col in cols:

        df[col] = df[col].apply(pd.to_numeric)

 
df.head()
df.groupby("Nom").mean()