# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns  # plot lib

import matplotlib.pyplot as plt  # plot lib



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
pkm_data = pd.read_csv('../input/Pokemon.csv')  # import data from file

pkm_data.head()                                 # print head of the table
type1_desc = pkm_data['Type 1'].describe()

print('Type 1 description')

print(type1_desc)

print()



type1_freq_norm = pkm_data['Type 1'].value_counts(sort=False, normalize=True)

print('Type 1 normalized discribution')

print(type1_freq_norm * 100)

print()



type1_freq = pkm_data['Type 1'].value_counts(sort=False)

print('Type 1 count distribution')

print(type1_freq)

print()
# Plot a simple histogram with binsize determined automatically

sns.countplot(x='Type 1', data=pkm_data)

#sns.distplot(d, kde=False, color="b", ax=axes[0, 0])