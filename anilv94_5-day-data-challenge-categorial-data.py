# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
responses = pd.read_csv("../input/anonymous-survey-responses.csv")
responses.describe()
responses.head()
reasonsOfInterest = responses["What's your interest in data science?"].value_counts()
labels = list(reasonsOfInterest.index)
# generate a list of numbers as long as our number of labels
positionsForBars = list(range(len(labels)))
plt.bar(labels,reasonsOfInterest.values)
plt.title("Reasons for interest in 5-Day data challenge")
import scipy.stats # statistics
statsInPast = responses["Have you ever taken a course in statistics?"].value_counts()
print(list(statsInPast.index))
statsInPast.values
scipy.stats.chisquare(statsInPast)
scipy.stats.chisquare(responses["Do you have any previous experience with programming?"].value_counts())
# now let's do a two-way chi-square test. Is there a relationship between programming background 
# and stats background?

contingencyTable = pd.crosstab(responses["Do you have any previous experience with programming?"],
                              responses["Have you ever taken a course in statistics?"])

scipy.stats.chi2_contingency(contingencyTable)
