# import our libraries

import scipy.stats # statistics

import pandas as pd # dataframe



# read in our data

surveyData = pd.read_csv("../input/anonymous-survey-responses.csv")
# first let's do a one-way chi-squared test for stats background

scipy.stats.chisquare(surveyData["Have you ever taken a course in statistics?"].value_counts())
# first let's do a one-way chi-squared test for programming background

scipy.stats.chisquare(surveyData["Do you have any previous experience with programming?"].value_counts())
# now let's do a two-way chi-square test. Is there a relationship between programming background 

# and stats background?



contingencyTable = pd.crosstab(surveyData["Do you have any previous experience with programming?"],

                              surveyData["Have you ever taken a course in statistics?"])



scipy.stats.chi2_contingency(contingencyTable)
## .016*3

.05/3

### therefore

0.016666666666666666*3
