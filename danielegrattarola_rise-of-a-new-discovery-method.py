# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Read data

data = pd.read_csv('../input/oec.csv')
# Familiarize with data

data.head()
# Group the data by discovery method and year

df = data.groupby(['DiscoveryMethod', 'DiscoveryYear']).size().unstack().T.fillna(0).astype(int)

df

# Create the columns for plotting

df = df.reset_index()
# Plot the number of discoveries for each method over time, both in normal and logarithmic scale

df.drop('DiscoveryYear', axis=1).plot(x=df['DiscoveryYear'], logy=False, title='# of discoveries per year').set_ylabel("# of discoveries")

df.drop('DiscoveryYear', axis=1).plot(x=df['DiscoveryYear'], logy=True, title='log # of discoveries per year').set_ylabel("log # of discoveries")