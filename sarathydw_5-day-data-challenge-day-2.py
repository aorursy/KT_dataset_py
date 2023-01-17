import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data=pd.read_csv("../input/sets.csv")

data.describe()
col1=data["year"]
plt.hist(col1,edgecolor = "black")

plt.title("Histogram of year column")

plt.xlabel("Year")

plt.ylabel("Count")