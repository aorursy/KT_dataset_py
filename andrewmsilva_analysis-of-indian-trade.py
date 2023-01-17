import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import_data = pd.read_csv("/kaggle/input/india-trade-data/2018-2010_import.csv")

export_data = pd.read_csv("/kaggle/input/india-trade-data/2018-2010_export.csv")
import_data.head()
export_data.head()
plt.figure(figsize= (15,5))

sns.lineplot(x='year',y='value', data=import_data, label='Imports')

sns.lineplot(x='year',y='value', data=export_data, label='Exports')

plt.title('Values of Indian imports and exports', fontsize=16)

plt.xlabel('Year')

plt.ylabel('Value in million US$')

plt.show()