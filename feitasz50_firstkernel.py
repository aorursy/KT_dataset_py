import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('../input/dataj.csv')

df.head()
df.info()
import matplotlib.pyplot as plt

import seaborn as sns
x = df["lon"]

y = df["lat"]
plt.figure(figsize=(10,10))

plt.title("Cities Latitude and Longitude")

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.scatter(x, y,s=1, alpha=1)

plt.show()