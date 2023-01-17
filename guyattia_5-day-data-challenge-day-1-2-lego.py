# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# Read data csv files into dataframes

colors = pd.read_csv('../input/colors.csv')

inventories = pd.read_csv('../input/inventories.csv')

inventory_parts=pd.read_csv('../input/inventory_parts.csv')

inventory_sets=pd.read_csv('../input/inventory_sets.csv')

part_categories=pd.read_csv('../input/part_categories.csv')

parts=pd.read_csv('../input/parts.csv')

sets=pd.read_csv('../input/sets.csv')

themes=pd.read_csv('../input/themes.csv')



#Describe data

print(sets.describe())



# Plot Histog

import matplotlib.pyplot as plt

sets_years=sets['year']

plt.hist(sets_years)

plt.title('Sets Years')