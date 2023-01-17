import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt



df = pd.read_csv('../input/cereal.csv')

df.describe()
plt.hist(df['calories'])

plt.title("Calories")

plt.show()
plt.hist(df['protein'])

plt.title("Protein")

plt.show()
plt.hist(df['fat'])

plt.title("Fat")

plt.show()