# This kernel will show you how to visualize a numeric column as a histogram.



# Load in libraries and packages

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# List files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Read data in to dataframe

df = pd.read_csv("../input/data_set_ALL_AML_independent.csv")



# Summarize data

df.describe()



# Select numeric column to analyze

x = df["39"]



# Plot histogram of column

sns.distplot(x, kde = False).set_title("Gene Marker 39")
