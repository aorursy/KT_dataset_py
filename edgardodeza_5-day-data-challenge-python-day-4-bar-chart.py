import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization

# change the plotting style
plt.style.use("fivethirtyeight")
digimon_movelist = pd.read_csv("../input/DigiDB_digimonlist.csv")
digimon_movelist.head()
print(digimon_movelist["Type"].unique())
digimon_movelist['Type'].value_counts().plot(kind='bar')

plt.title('Digimon types')
plt.ylabel('Count')
plt.show()
digimon_movelist['Type'].value_counts(sort=False).plot(kind='bar')

plt.title('Digimon types')
plt.ylabel('Count')
plt.show()