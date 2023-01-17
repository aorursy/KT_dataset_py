# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
working_file = "/kaggle/input/covid19-confirmed-cases-in-bangladesh/BD_COVID19_data.csv"
data = pd.read_csv(working_file)
data.head()
data.shape
data.describe()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.distplot(data['Total_Cases'],kde=False, bins=30)
data['Division'].value_counts()
from matplotlib.pyplot import figure 
plt.figure(figsize=(15,10))
sns.boxplot(x=data['Division'],y=data['Total_Cases'])
table = pd.pivot_table(data,values='Total_Cases',index=['Division'])
table.sort_values(by=['Total_Cases'],ascending=False)
# This is for creating seperate DataFrames for each Divisions
# It'll create 8 new dataframes e.g. Dhaka will be Dhaka_data
Division_names = list(set(data['Division']))
for name in Division_names:
    globals()[name+"_data"] = data[data['Division']==name]
    
    
Dhaka_data
# We can create pivot table on each Division seperatly
pv_dhaka = pd.pivot_table(Dhaka_data, values='Total_Cases', index=['Districts'])
pv_dhaka.sort_values(by=['Total_Cases'],ascending=False)
