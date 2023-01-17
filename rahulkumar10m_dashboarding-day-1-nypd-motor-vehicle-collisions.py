import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/nypd-motor-vehicle-collisions.csv')
dataset.head()
dataset.tail()
dataset.info()
dataset.describe()
# Filtering the dataframe per BOROUGH
injured_people = dataset[dataset['NUMBER OF PERSONS INJURED'] > 0]
killed_people = dataset[dataset['NUMBER OF PERSONS KILLED'] > 0]

# Plotting the dataframe
fig = plt.figure(figsize=(15, 10))

ax = fig.add_subplot(111)
ax2 = ax.twinx()
injured_people.BOROUGH.value_counts().plot(kind='bar', width = 0.4, color='blue', position=0, ax=ax)
killed_people.BOROUGH.value_counts().plot(kind='bar', width = 0.4, color='red', position=1, ax=ax2)
ax.set_title('Number of persons injured and killed per borough', fontsize=20, fontweight='bold')
ax.set_ylabel('Number of persons injured')
ax2.set_ylabel('Number of persons killed')
plt.show()