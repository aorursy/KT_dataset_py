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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import pandas_profiling
import string
from IPython.display import display
import plotly.graph_objs as go
import plotly.express as px
import plotly
plotly.offline.init_notebook_mode(connected=True)


df = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
"""

Paid vs Free


"""
free = len(df[df.Type == 'Free'])
paid = len(df[df.Type == 'Paid'])
print('There are {} free apps in the dataset'.format(free))
print('There are {} paid apps in the dataset'.format(paid))
"""

Most popular category


"""
#Removing unwanted values
#noneed = df.loc[df.Size == 'Varies with device']
#noneed.drop(noneed.index, inplace=True)

#Removing symbols to create only numerics column
df.Size = df.Size.apply(lambda x: str(x).replace('+', ''))
df.Size = df.Size.apply(lambda x: str(x).replace('M', ''))
df.Size = df.Size.apply(lambda x: str(x).replace('k',''))

# Convert size column in numeric
df.Size = pd.to_numeric(df.Size, errors='coerce')
result = df.groupby('Category')['Size'].sum().reset_index()
final_result = result.sort_values('Size', ascending=False).reset_index(drop=True)

# Visualization

final_result.set_index('Category', inplace=True)
final_result.plot(kind='bar', figsize=(20,11), color='green')
plt.xlabel('Category')
plt.ylabel('Size')
"""

App with the largest size


"""

print(df.sort_values('Size', ascending=False).iloc[0]['App'])


"""

App with large number of reviews


"""
print(df.sort_values('Reviews', ascending=False).iloc[0]['App'])
"""

Clean data


"""
df.dropna(subset=['Rating'], inplace=True)
df.dropna(subset=['Size'], inplace=True)
df.dropna(subset=['Android Ver'], inplace=True)
df.dropna(subset=['Current Ver'], inplace=True)

df = df.drop_duplicates().reset_index(drop=True)


# Let's check duplicates
df.duplicated().sum()
# Let's check 0 values
df.isnull().sum()
"""

Data visualization


"""

report = pandas_profiling.ProfileReport(df) # kinda handy tool to understand data we are working on and this will save your time on EDA process.
display(report) 
sns.set_style('darkgrid')
fig,(ax1) = plt.subplots(1,figsize=(20,11))
plt.suptitle('Count plots')
sns.countplot(y='Android Ver', data=df, ax=ax1)
plt.show()
fig,(ax2) = plt.subplots(figsize=(20,11))
plt.suptitle('Count plots')
sns.countplot(y='Category', data=df, ax=ax2)
plt.show()
fig,(ax3) =  plt.subplots(figsize=(20,11))
plt.suptitle('Count plots')
sns.countplot(y='Installs', data=df, ax=ax3)
plt.show()
tot_num_of_apps_in_category = df.Category.value_counts().sort_values(ascending=True)
data = [go.Pie(
        labels = tot_num_of_apps_in_category.index,
        values = tot_num_of_apps_in_category.values,
        hoverinfo = 'label+value'
)]

plotly.offline.iplot(data, filename='active_category')
