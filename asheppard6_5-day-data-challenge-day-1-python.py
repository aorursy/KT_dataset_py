# This kernel will show you how to read data into a dataframe and summarize the data in python



# Load in packages

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# List the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Read data into dataframe using pandas read_csv() function

df = pd.read_csv("../input/survey.csv")



# Summarize data

df.describe()