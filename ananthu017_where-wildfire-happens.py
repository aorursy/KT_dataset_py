# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Plotting the csv data



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load the Wildfire data

wildfire_list = pd.read_csv('/kaggle/input/california-wildfire-incidents-20132020/California_Fire_Incidents.csv')



# Select Latitude and Longitude values for the wildfire location

wildfire_list = wildfire_list[['Name','Latitude', 'Longitude']].set_index('Name')



# Drop columns with no data

wildfire_list = wildfire_list.dropna()

wildfire_list.head()
# For California, Latitude range - (32.5121, 42.0126) and Longitude range - (-124.6509, -114.1315)

# Select the data belonging to the above range

wildfire_list = wildfire_list[(wildfire_list['Latitude'] > 32.5121) & (wildfire_list['Latitude'] < 42.0126)]

wildfire_list = wildfire_list[(wildfire_list['Longitude'] > -124.6509) & (wildfire_list['Longitude'] < -114.1315)]
# Plot the lat and long data to identify the region where wildfire is frequent

plt.figure(figsize=(18,8))

plt.scatter(wildfire_list['Latitude'], wildfire_list['Longitude'], marker='x', color='firebrick')

plt.xticks(np.arange(min(wildfire_list['Latitude']), max(wildfire_list['Latitude'])+0.2, 0.2), rotation=90)

plt.yticks(np.arange(min(wildfire_list['Longitude']), max(wildfire_list['Longitude'])+0.3, 0.3))

plt.grid()

plt.xlabel('Latitude Range', fontsize=12)

plt.ylabel('Longitude Range', fontsize=12)

plt.title('WildFires (2013 - 2019)', fontsize=16)