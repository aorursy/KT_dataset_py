# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('default')
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
carCsv = pd.read_csv('../input/cardataset/data.csv')
carCsv.columns = carCsv.columns.str.lower()
# Create filter to setup dataframes for midrange cars
def rangeFilter(make):
    return carCsv[(carCsv['msrp'] > 30000) & (carCsv['msrp'] < 100000) & 
           (carCsv['make'] == make) &
           (carCsv['market category'].notna())].drop(
        columns = ['popularity', 'number of doors'])
df_toyota = rangeFilter('Toyota')
df_lexus = rangeFilter('Lexus')
df_scion = rangeFilter('Scion')
df_acura = rangeFilter('Acura')
df_honda = rangeFilter('Honda')

# Dataframes for parent companies
toyota_df = [df_toyota, df_lexus, df_scion]
honda_df = [df_acura, df_honda]
# Plot some examples of midrange Asian cars
def plot_hist(df_group):
    for df in df_group:
        sns.distplot(a=df['msrp'], label=df['make'], kde=False)
    plt.title('MSRP Comparison')
    plt.ylabel('# of models')
    plt.xlabel('MSRP')
    plt.legend()
plot_hist(toyota_df)
plot_hist(honda_df)
carMakes = set(carCsv['make'])

# Create a dictionary of car makes and median price
# Sort by median price
median_make_d = {}
for carMake in carMakes:
    make = carCsv.loc[carCsv['make'] == carMake]
    # A limit of >15k seems like the right limit to ignore outliers on the lower end    
    median_make = make['msrp'][make['msrp'] > 20000].median()
    median_make_d[median_make] = carMake
sorted_median_make_d = dict(sorted(median_make_d.items(), reverse=True))

median_make_df = pd.DataFrame(sorted_median_make_d.items(), columns = ['median_msrp', 'make'])
plt.figure(figsize=(5,10))
sns.barplot(x='median_msrp', y='make', data=median_make_df)
# Removing Bugatti to have better visibility in graph
from copy import deepcopy

# Create deep copy to avoid changing `sorted_median_make_d`  
no_bugatti_median_d = deepcopy(sorted_median_make_d)
bugatti_key = max(no_bugatti_median_d.keys())
no_bugatti_median_d.pop(bugatti_key)

no_bugatti_df = pd.DataFrame(no_bugatti_median_d.items(), columns = ['median_msrp', 'make'])
plt.figure(figsize=(15,16))
plt.title('Median MSRP - 2017 & older models')
bp = sns.barplot(x='median_msrp', y='make', data=no_bugatti_df)

# Display value on the graph
# Functionality is available on matplotlib
# Get a variable from matplotlib
ax = plt.gca()

# Use patches() and text() to annotate bars
for p in ax.patches:
    width = p.get_width()
    height = p.get_y() + p.get_height()/2. +0.2
    val = int(width)
    
    # Using comma as thousands separator
    formatted_val = f'{val:,}' 
    
    ax.text(width, 
            height,
            formatted_val,
            color='black',
            ha='left')
