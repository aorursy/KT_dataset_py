# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/crime_homicide_subset.csv', encoding='latin1', sep=',')
df.info()
g = sns.factorplot(x='OFFENSE', data=df, kind='count', size=6)

g.set_axis_labels('Offense', 'Number of Crimes')

g.set_xticklabels(rotation=90)
g = sns.factorplot(x="year", hue="OFFENSE", data=df, size=6, kind="count")

g.set_axis_labels('Year', 'Number of Crimes')
g = sns.factorplot(x='mont', data=df, kind='count', size=6)

g.set_axis_labels('Month', 'Number of Crimes')
g = sns.factorplot(x='year', hue="SHIFT", data=df, kind='count', size=6)

g.set_axis_labels('Year', 'Number of Crimes')
g = sns.factorplot(x='OFFENSE', hue="week", data=df, kind='count', size=6)

g.set_axis_labels('Offense', 'Number of Crimes')