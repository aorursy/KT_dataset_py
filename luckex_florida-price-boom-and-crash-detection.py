# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Load the file

df = pd.read_csv('../input/HPI_master.csv')



# Describe dataset qnd bqsic statistcs

#print(df.head())

#print(df.tail())

print("Dataframe size:\n", df.count())



for x in df.columns:

    print(x,df[x].unique())



print(df.describe())
import matplotlib.pyplot as plt



plt.style.use('ggplot')

%matplotlib inline



fig, ax = plt.subplots(figsize=(6,4))

ax.hist(df['yr'], color='black')

ax.set_ylabel('Count per Year', fontsize=12)

ax.set_xlabel('Year', fontsize=12)

plt.title('Number of points per year', fontsize=14, y=1.01)
fig, ax = plt.subplots(figsize=(6,6))

ax.scatter(df['yr'], df['index_nsa'], color='green')

ax.set_xlabel('Year', fontsize=12)

ax.set_ylabel('Price index', fontsize=12)

plt.title('Price index evolution over year', fontsize=14, y=1.01)