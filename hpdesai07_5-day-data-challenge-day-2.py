import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



data = pd.read_csv('../input/archive.csv')

dataframe = data.describe()

plt.hist(dataframe["February Average Temperature"])

plt.title("February Data")