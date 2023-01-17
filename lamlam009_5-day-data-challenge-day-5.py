# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Health_AnimalBites.csv")

df.head()
df = df.dropna(subset=["GenderIDDesc", "WhereBittenIDDesc"])

df = df[(df["GenderIDDesc"] != 'UNKNOWN')]

df = df[(df["WhereBittenIDDesc"] != 'UNKNOWN')]

df = df[df["SpeciesIDDesc"] == 'DOG']
scipy.stats.chisquare(df["GenderIDDesc"].value_counts())
scipy.stats.chisquare(df["WhereBittenIDDesc"].value_counts())
contingencyTable = pd.crosstab(df["GenderIDDesc"], df["WhereBittenIDDesc"])

contingencyTable
scipy.stats.chi2_contingency(contingencyTable)