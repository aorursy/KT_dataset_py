#import our libraries 

import scipy.stats # statistics 

import pandas as pd # dataframe 



# read in our data 

surveyData = pd.read_csv("../input/anonymous-survey-responses.csv")
surveyData.head()
# first let's do a one way chi squared test for stats background 

scipy.stats.chisquare(surveyData["Have you ever taken a course in statistics?"].value_counts())
scipy.stats.chisquare(surveyData["Do you have any previous experience with programming?"].value_counts())
# doing a two way test

contingencyTable = pd.crosstab(surveyData["Do you have any previous experience with programming?"], surveyData["Have you ever taken a course in statistics?"])
contingencyTable
scipy.stats.chi2_contingency(contingencyTable)