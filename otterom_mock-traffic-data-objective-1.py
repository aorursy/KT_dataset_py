# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from numpy import NaN # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import re

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="whitegrid")

plt.close('all')



df = pd.read_csv('/kaggle/input/mock-traffic-data/MockTrafficDataForMCNFP.csv')



# Drop `id` and `plate_id`

df.drop(['id', 'plate_id'], axis=1, inplace=True)
# View sample of updated data frame

df.head(10)
# Get list of node columns from dataframe

# Don't really need this, but it might be handy down the line.

node_cols: list = [i for i in df.columns if i.startswith('time_node')]



### We're going to clean the data by replacing any null values with zero and 

### any valid time values with 1 in our node columns

df.loc[:, node_cols] = df[node_cols].replace(pd.NaT, 0,)

df.loc[:, node_cols] = df[node_cols].where(df[node_cols] == 0, 1)
# View sample of updated data frame

df.head(10)
# Get the sum for each time_node

series_obj_1 = df[node_cols].sum().astype(int)



# Convert series to dataframe

df_obj_1 = series_obj_1.to_frame(name='Count')



# Add time_node column and reset the index

df_obj_1['Node_Names'] = df_obj_1.index

df_obj_1.reset_index(drop=True, inplace=True)
# Plot the value counts

fig, ax1 = plt.subplots(figsize=(9, 7))

sns.barplot(x='Count', y='Node_Names', data=df_obj_1, ci='sd', palette='Blues_d', ax=ax1)

ax1.set_ylabel('Nodes')

ax1.set_xlabel('Count')

ax1.set_title('Frequency of Plates through Nodes')

plt.tight_layout()

plt.show()