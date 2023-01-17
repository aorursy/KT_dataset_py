import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn

import matplotlib.pyplot as plt
df = pd.read_csv('../input/labels.csv')

freq = df['breed'].value_counts()
plt.figure()

seaborn.barplot(x=freq.index, y=freq)

plt.xticks(rotation=90)

plt.title('Breed Distribution: Sorted')

plt.show(block=False)
plt.figure()

freq_sorted = freq.sort_index()

seaborn.barplot(x=freq_sorted.index, y=freq_sorted)

plt.xticks(rotation=90)

plt.title('Breed Distribution: Alphabetically')

plt.show(block=False)