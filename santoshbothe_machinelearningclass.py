# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
url = "../input/WorldCups.csv"
names = ['Year', 'Country', 'Winner','Runners-Up',
'Third', 'Fourth', 'GoalsScored', 'QualifiedTeams', 'MatchesPlayed','Attendance']
dataset = pandas.read_csv(url, names=names)

# shape
print(dataset.shape)
# head
print(dataset.head(20))
# descriptions