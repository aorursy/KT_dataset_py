# Start by importing the basics



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# Read the csv file through pandas and summarise the data

# Obs. The 'include = 'all'' guarantees all columns (numeric or not) are summarised



temp_df = pd.read_csv("../input/GlobalTemperatures.csv")

pd.DataFrame.describe(temp_df, include = 'all')



# Show the first few lines of the dataset

temp_df.head()

# Get the column that shows all data for average land temperature

landTemp = temp_df["LandAverageTemperature"]



plt.hist(landTemp.dropna())

# from stack overflow: the dropna gets rid of NaN values! Otherwise the histogram won't plot.

plt.title("Average of global land temperatures (since 1750)")

plt.xlabel("Temperature (oC)")

plt.ylabel("Frequency")


