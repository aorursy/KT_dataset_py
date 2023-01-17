# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# import our libraries

import matplotlib.pyplot as plt



# read in our data

nutrition = pd.read_csv('../input/starbucks_drinkMenu_expanded.csv')

nutrition.describe().transpose()



# nutrition.describe(include='all') # this will shwo all columns including non-numberic data  
# list all the column names



nutrition.columns



# get the sodium colum

sodium = nutrition[' Sodium (mg)']

# plot a histogram of sodium content with nine bins and black edge color

plt.hist(sodium, bins=9, edgecolor='black')

plt.title('Sodium in Starbucks Items') # add a title

plt.xlabel('Sodium in milligrams') # label the x axes

plt.ylabel('Count') # label the y axes
# another way of plotting a histogram (from the pandas plotting API)

nutrition.hist(column= ' Sodium (mg)', bins=9, edgecolor='black', figsize=(8,5))



plt.title('Sodium in Starbucks Items') # add a title

plt.xlabel('Sodium in milligrams') # label the x axes

plt.ylabel('Count') # label the y axes