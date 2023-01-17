import os
import json

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
%matplotlib inline
print(os.listdir("../input"))

COLLISIONS_CSV = '../input/nypd-motor-vehicle-collisions.csv'
METADATA_JSON = '../input/socrata_metadata.json'
DATADICT = '../input/Collision_DataDictionary.xlsx'
collisions = pd.read_csv(COLLISIONS_CSV)
collisions.head()
collisions.tail()
collisions.info()
print(f'Number of columns: {len(collisions.columns)}')
collisions.describe()
# Assuming that the DATE and Time columns are sorted (meaning row 0 is latest date and last row represents the earliest date)
latest_date = collisions.iloc[0].DATE
earliest_date = collisions.iloc[-1].DATE
latest_date
earliest_date
# Filter the dataframes into number of injured and number of deaths
injured_df = collisions[collisions['NUMBER OF PERSONS INJURED'] > 0]
killed_df = collisions[collisions['NUMBER OF PERSONS KILLED'] > 0]
# Now plot these dataframes per borough
fig, ax = plt.subplots(1, figsize=(12,8))
injured_df.BOROUGH.value_counts().plot(kind='bar', color='blue', position=0, ax=ax)
killed_df.BOROUGH.value_counts().plot(kind='bar', color='red', position=1, ax=ax)
ax.set_title('Number of persons injured and killed per borough', fontsize=20, fontweight='bold')
ax.set_ylabel('Number of persons injured/killed')
plt.show()
fig = plt.figure(figsize=(12,8))

bx = fig.add_subplot(111)
bx2 = bx.twinx() # Create another axes that shares the same x-axis as bx.
injured_df.BOROUGH.value_counts().plot(kind='bar', color='blue', width=0.4, position=0, ax=bx)
killed_df.BOROUGH.value_counts().plot(kind='bar', color='red', width=0.4, position=1, ax=bx2)
bx.set_title('Number of persons injured and killed per borough', fontsize=20, fontweight='bold')
bx.set_ylabel('Number of persons injured')
bx2.set_ylabel('Number of persons killed')

plt.show()
