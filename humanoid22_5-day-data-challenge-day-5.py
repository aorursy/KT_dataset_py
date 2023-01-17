# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats

#import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data = pd.read_csv("../input/database.csv", low_memory=False)



airport = data["Airport"]

flightPhase = data["Flight Phase"]



print("Airport Chi-square test")

print(scipy.stats.chisquare(airport.value_counts()))



print("Flight Phase Chi-square test")

print(scipy.stats.chisquare(flightPhase.value_counts()))



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
sns.countplot(x=airport, order=airport.value_counts().iloc[:3].index)


sns.countplot(x=flightPhase, order=flightPhase.value_counts().iloc[:5].index)