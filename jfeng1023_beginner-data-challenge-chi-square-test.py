# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

aircraft = pd.read_csv("../input/database.csv")
aircraft.head()

from scipy.stats import chisquare, chi2_contingency as chi2

# One-way chisquare test to see if states are evenly distributed
chisquare(aircraft['State'].value_counts())

# Combine the "engine shutdown" and "engine shut down" entries in "flight impact" column
aircraft.loc[aircraft['Flight Impact']=='ENGINE SHUT DOWN','Flight Impact'] = 'ENGINE SHUTDOWN'

# Two-way chisquare test for the relationship between visibility and flight impact
contingencyTable = pd.crosstab(aircraft['Visibility'],aircraft['Flight Impact'])
print(contingencyTable)
chi2(contingencyTable)

# Two-way chisquare test to see whether mourning doves and gulls cause different impact
subset = aircraft.loc[aircraft['Species Name'].isin(['MOURNING DOVE','GULL'])]

bird_impact = pd.crosstab(subset['Species Name'],subset['Flight Impact'])
chi2(bird_impact)
