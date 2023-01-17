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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Read the input of both CSV files

layout_data = pd.read_csv('../input/public_layout.csv', sep=',')

data = pd.read_csv('../input/recs2009_public.csv', sep=',')
# grab the info about layout data

layout_data.info()
data.info()
layout_data.head(5)
data.head(5)
columns = data.columns

print('Length of data columns: ',  len(columns))
print("Number of different variables: " + str(layout_data['Variable Name'].nunique()))
# create the variable names and variable descriptions in two separate lists: from dataColumns

labels = list(layout_data['Variable Label'])

places = list(layout_data['Variable Name'])



# Creating a function to find the variable definition of any variable

def whatIs(place):

    pl = places.index(place)

    print(place, ' is ', labels[pl])


whatIs("DOEID")

whatIs("DIVISION")

whatIs("REGIONC")

whatIs("REPORTABLE_DOMAIN")

whatIs("TYPEHUQ")

whatIs("NWEIGHT")

whatIs("HDD65")

whatIs("CDD65")

whatIs("HDD30YR")

whatIs("CDD30YR")

whatIs("SCALEEL")

whatIs("KAVALNG")

whatIs("PERIODNG")

whatIs("SCALENG")

whatIs("PERIODLP")

'''

# We could also define them all with a for loop

for i in places:

    whatIs(str(i))

    '''

# Starting to feel bad for the people who had to analyze this data
# create scatterplots! 

# This one is total KWH, total Electricity cost, Electricity per site, and age of resident

# There are thousands of comparisons! 

# with all 931 variables, and if each one got a unique parter, there is 432,915 unique combos

# if all 931 variables are used with each other 931, that grows to: 8.000111451E+57 unique combos

# twice as many stars in the universe

sns.set()

cols = ['KWH', 'DOLLAREL', 'BTUEL', 'HHAGE']



''' # Dont do this lol

for i in places:

    cols.append(i)

'''

sns.pairplot(data[cols], size = 1.9)

plt.show()
# I like the KWH structure, lets dive into that once more

sns.set()

whatIs('DOLLAREL')

whatIs('KWH')

rowCol = ['KWH', 'DOLLAREL']

sns.regplot(x="KWH", y="DOLLAREL", data=data[rowCol])

plt.show()
# Lets make another, instead using BTUEL

whatIs('BTUEL')

rowCol = ['KWH', 'BTUEL']

sns.regplot(x="KWH", y="BTUEL", data=data[rowCol])

plt.show()
# Now lets have some fun with YEARMADE and HHAGE

whatIs('YEARMADE')

whatIs('HHAGE')

variable = 'YEARMADE'

data = pd.concat([data['HHAGE'], data[variable]], axis=1)

f, ax = plt.subplots(figsize=(13, 8))

fig = sns.boxplot(x=variable, y="HHAGE", data=data)

fig.axis(ymin=0, ymax=110);

plt.xticks(rotation=90);