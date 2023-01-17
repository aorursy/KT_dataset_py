# Load packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Load file

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        file = os.path.join(dirname, filename)
data = pd.read_csv(file)
print(f"Number of rows: {len(data)}")
print(f"Columns: {data.columns.tolist()}")


this_year = 2020
plt.figure(figsize = (10,7));
plt.hist(data['release_year'], bins = np.arange(1940,this_year+1, 1));
plt.hist(data['release_year'][data['release_year']==this_year], bins = np.arange(this_year,this_year+2, 1), label="Incomplete");
plt.yscale('log');
plt.xlabel("Year");
plt.ylabel("Number of pictures on Netflix");
plt.title("Number of pictures on Netflix per release year");
plt.legend();