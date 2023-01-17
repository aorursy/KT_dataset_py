# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns 

import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore')
df = pd.read_csv('/kaggle/input/icc-test-cricket-runs/ICC Test Batting Figures.csv' ,  encoding = "ISO-8859-1")

df.shape
df.drop('Player Profile' , axis = 1 , inplace = True)
df_new = df[df['Mat'] > 50]

df_new['Avg'] = pd.to_numeric(df_new['Avg'])

df_new.sort_values('Avg' , ascending = False , inplace = True)

df_plot = df_new.head(10)
plt.figure(figsize=(20,5))

plt.bar(df_plot['Player'] , df_plot['Avg'])

plt.xlabel('Players')

plt.ylabel('Average')

plt.title('Players vs Average (Top 10)')
df_new['Mat'] = pd.to_numeric(df_new['Mat'])

df_new.sort_values('Mat' , ascending = False , inplace = True)

df_plot = df_new.head(10)

df_new.head(10)
plt.figure(figsize=(20,5))

plt.bar(df_plot['Player'] , df_plot['Mat'])

plt.xlabel('Players')

plt.ylabel('Number of Matches played')

plt.title('Players vs Number of matches played (Top 10)')
df_new['100'] = pd.to_numeric(df_new['100'])

df_new.sort_values('100' , ascending = False , inplace = True)

df_plot = df_new.head(10)

df_new.head(10)
plt.figure(figsize=(20,5))

plt.bar(df_plot['Player'] , df_plot['100'])

plt.xlabel('Players')

plt.ylabel('Number of 100s')

plt.title('Players vs Number of 100s (Top 10)')
df_new['50'] = pd.to_numeric(df_new['50'])

df_new.sort_values('50' , ascending = False , inplace = True)

df_plot = df_new.head(10)

df_new.head(10)
plt.figure(figsize=(20,5))

plt.bar(df_plot['Player'] , df_plot['50'])

plt.xlabel('Players')

plt.ylabel('Number of 50s')

plt.title('Players vs Number of 50s (Top 10)')
df_new['HS'] = df_new['HS'].str.replace('*' , '')

df_new['HS'] = pd.to_numeric(df_new['HS'])

df_new.sort_values('HS' , ascending = False , inplace = True)

df_new = df_new.head(10)
plt.figure(figsize = (22,5))

plt.bar(df_new['Player'] , df_new['HS'])

plt.xlabel('Players')

plt.ylabel('Highest batting scores')

plt.title('Plot of Top 10 Individual highest batting score')