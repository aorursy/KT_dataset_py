import scipy.stats # statistics 



import numpy as np # linear algebra

import pandas as pd #dataframe



surveyData = pd.read_csv('../input/anonymous-survey-responses.csv')
surveyData.head()
# first, perform a one-way chi-squared test for stats background

scipy.stats.chisquare(surveyData['Have you ever taken a course in statistics?'].value_counts())
# a two-way chi-square test. Is there a relationship between programming background 

# and stats background?



contingencyTable = pd.crosstab(surveyData['Do you have any previous experience with programming?'], 

                surveyData['Have you ever taken a course in statistics?'])

scipy.stats.chi2_contingency(contingencyTable)
contingencyTable