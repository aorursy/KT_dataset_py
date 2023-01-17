# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
dataset = pd.read_csv('../input/MOP_installed_capacity_sector_mode_wise.csv')
dataset.head()
dataset['Mode'].unique()
dataset['State'].unique()
sns.countplot(x = 'Mode',data = dataset)
dataset[dataset['Installed Capacity'] == max(dataset['Installed Capacity'])]
state = dataset.groupby(by = 'State').sum().reset_index()
state
state.max()
fig_size = plt.figure(figsize = (50,20))

plt.rcParams.update({'font.size':35})

sns.barplot(x = 'State',y = 'Installed Capacity',data=state,palette='coolwarm')

plt.xlabel('State')

plt.ylabel('Installed Capacity')
state_type = dataset.groupby(by = ['State','Mode']).sum().reset_index()
state_type.head()
state_type[state_type['Mode'] == 'RES'].sort_values(by = 'Installed Capacity',ascending = False)[:3]
state_type[state_type['Mode'] == 'Hydro'].sort_values(by = 'Installed Capacity',ascending = False)[:3]
state_type[state_type['Mode'] == 'Thermal'].sort_values(by = 'Installed Capacity',ascending = False)[:3]
state_type[state_type['Mode'] == 'Nuclear'].sort_values(by = 'Installed Capacity',ascending = False)[:3]
mode_based = dataset.groupby(by = 'Mode').sum().reset_index()
mode_based
plt.rcParams.update({'font.size':14})

plt.title('Power Generation Modes Distribution')

plt.pie(x = mode_based['Installed Capacity'],labels = mode_based['Mode'],

        colors = ['#65c6c4','#cf3030','#c9f658','#ff5a00'],explode = (0.0,0.0,0.1,0.0),shadow = True,autopct='%1d%%',

        startangle = 90)

plt.axis('equal')