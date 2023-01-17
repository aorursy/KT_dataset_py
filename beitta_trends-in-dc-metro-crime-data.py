# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

%matplotlib inline
# Load and examine the dataset

#df = pd.read_csv('../input/crime_homicide_subset.csv', encoding='latin1', sep=',')

#df.head(5)

#help('modules')



import os

#os.popen('pip freeze').readlines()

#os.popen('df -hl').readlines()

#os.popen('uname -a').readlines()

#os.popen('cat /proc/version').readlines()

#os.popen('free -g').readlines()

os.popen('cat /proc/cpuinfo').readlines()
# Examine available columns

df.columns.tolist()
g = sns.factorplot(x='year', data=df, kind='count', size=6)

g.set_axis_labels('Year', 'Number of Crimes')
g = sns.factorplot(x='mont', data=df, kind='count', size=6)

g.set_axis_labels('Month', 'Number of Crimes')
g = sns.factorplot(x='OFFENSE', data=df, kind='count', size=6)

g.set_axis_labels('Offense', 'Number of Crimes')

g.set_xticklabels(rotation=90)
g = sns.factorplot(x="year", hue="OFFENSE", data=df, size=6, kind="count")

g.set_axis_labels('Year', 'Number of Crimes')
g = sns.factorplot(x='year', hue="SHIFT", data=df, kind='count', size=6)

g.set_axis_labels('Year', 'Number of Crimes')