import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats 



df = pd.read_csv("../input/anonymous-survey-responses.csv")
#First we do a oneway chi Squared Test for statistic background

scipy.stats.chisquare(df['Have you ever taken a course in statistics?'].value_counts())
#First we do a oneway chi Squared Test for programming background

scipy.stats.chisquare(df['Do you have any previous experience with programming?'].value_counts())
contigencytable = pd.crosstab(df['Do you have any previous experience with programming?'],

                              df['Have you ever taken a course in statistics?'])
scipy.stats.chi2_contingency(contigencytable)