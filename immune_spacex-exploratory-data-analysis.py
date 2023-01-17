# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print(os.listdir('../input'))
import pandas as pd

# Create a variable named df for dataframe

df = pd.read_csv('../input/spacex-missions/database.csv')
# Print the shape of the dataset 

df.shape
# Print the size of the dataframe

df.size
# This will print the dataframe 🖼️

df
df.loc[0]
# Check for data types

df.dtypes
# Description of data

df.describe()
# Check dataframe for null values

df.isnull().sum()
# Delete null values from dataframe

df = df.dropna()
f, ax = plt.subplots(figsize=(10, 8))

corr = df.corr()

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)