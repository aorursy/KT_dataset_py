# Importing pandas to read csv file

import pandas as pd



# Input data files are available in the read-only "../input/" directory

# Lets check the path of the input files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Reading the file using pandas

file_path = "/kaggle/input/bangla-song-lyrics/BanglaSongLyrics.csv"

dataframe = pd.read_csv(file_path)
# Summary of the dataframe

dataframe.describe()
# Check a single value

index = 10

print(f"Title: {dataframe.title[index]}")

print(f"Category: {dataframe.category[index]}")

print(dataframe.lyrics[index])