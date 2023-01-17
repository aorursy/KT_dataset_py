import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv('../input/Pokemon.csv')

df.head(3)
df.iloc[[df[u'Total'].idxmax()]]
df.iloc[[df[u'Total'].idxmin()]]
df.iloc[[df[u'Defense'].idxmax()]]
df.iloc[[df[u'Attack'].idxmax()]]
# Create a pyplot figure big enough for all our pokemon

plt.figure(figsize=(12,8))

# Create the colors for the plot

colors = pd.DataFrame(df['Type 1'].astype('category')).apply(lambda x: x.cat.codes)

print(colors.head())



# Plot the pokemon Attack and Defense as points

plt.scatter(df['Attack'], df['Defense'],c=colors, cmap=plt.cm.prism, s=120)

plt.xlim(0,200)

plt.ylim(0,250)

plt.xlabel('Attack')

plt.ylabel('Defense')

plt.title('Pokemon Attack, Defense, and their Types')