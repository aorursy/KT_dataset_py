import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

df = pd.read_csv("../input/Air_Quality_Survey.csv")

df.head(5)
plt.figure(figsize=(10,6))

df.Parameter.value_counts()[:10].plot(kind="bar")

plt.xticks(rotation=60)

plt.ylabel("Number of Times Measured")