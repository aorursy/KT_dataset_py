# Loading Libraries

import numpy as np

import pandas as pd
# Read CSV file by panda

df = pd.read_csv("../input/database.csv",low_memory=False)

df.describe()
# Plot a Numeric Variable with a Histogram

import matplotlib.pyplot as plt
df["Incident Year"].hist()

plt.title('Incident Year v/s Damage Frequency')

plt.xlabel("Incident Year")

plt.ylabel("Damage frequency")

# Let's see in which year most of the damages happened by birds