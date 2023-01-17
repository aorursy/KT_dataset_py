import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib



horses = pd.read_csv("../input/horses.csv")

print(horses.dtypes)



# Descriptive

# Total horses

print("####")

print("Head")

print("####")



print(horses.head())
# Count the number of each sire, I'm selecting the 'age' column to return

# counts but any column would return counts, there is probably a better

# way to accomplish this in pandas. Let me know if you know.

sire_group = horses.groupby('sire_id')

dam_group = horses.groupby('dam_id')





counts = dam_group.count()