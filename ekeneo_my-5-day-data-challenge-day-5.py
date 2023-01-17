# import our libraries

import scipy.stats # Statistics

import pandas as pd # dataframe



# Read in our data

surveyData = pd.read_csv('../input/anonymous-survey-responses.csv')
# first let's do a one-way chi-squared test for stats background



scipy.stats.chisquare(surveyData["Have you ever taken a course in statistics?"].value_counts())
# let's do a one-way chi-squared test for programming background

scipy.stats.chisquare(surveyData["Do you have any previous experience with programming?"].value_counts())
# Now let's do a two-way chi-square test. Is there a relationship between programming background 

# and stats background?



contigenctTable = pd.crosstab(surveyData["Have you ever taken a course in statistics?"],

                              surveyData["Do you have any previous experience with programming?"])



scipy.stats.chi2_contingency(contigenctTable)

# import seaborn and alias it as sns

import seaborn as sns



# To make a barplot from a columns in our dataframe use:

# sns.countplot(surveyData["Have you ever taken a course in statistics?"],palette="cool")



# make a barplot from two categorical columns in our dataframe(surveyData)

sns.countplot(surveyData["Have you ever taken a course in statistics?"],

              hue=surveyData["Do you have any previous experience with programming?"], palette="Set2")