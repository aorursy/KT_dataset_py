import matplotlib

import matplotlib.pyplot as plt

import pandas as pd



df = pd.read_csv('../input/crimes.csv')



df.groupby('Offense').size().sort_values(ascending=False).plot(kind='bar')

plt.xlabel('Offense')

plt.ylabel('Total')

plt.title('Offenses and Their Frequency')