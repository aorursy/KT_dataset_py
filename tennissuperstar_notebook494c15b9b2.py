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
train = pd.read_csv('../input/train.csv')
print(train.head)

print(train.columns[0:20])

print(train.columns[20:40])

print(train.columns[40:60])

print(train.columns[60:80])

print(train.columns[80:100])

print(train.columns[100:120])

print(train.columns[120:])

#print(train.shape)
# This tells me that there are 188318 samples and 132 different variables for each

print(train.shape)
#Print all rows and columns. Dont hide any

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)



print(train.describe())
print(train.skew())
import matplotlib.pyplot as plt

import seaborn as sns



# Create a 2x4 grid using subplot

for row in range(0,2):

    f, ax = plt.subplots(nrows=1, ncols=4, figsize=(10,10))

    for col in range(0,4):

        sns.violinplot(train, ax=ax[col])
