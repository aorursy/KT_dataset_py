import pandas as pd # data processing

import scipy.stats # statistics

df = pd.read_csv("../input/anonymous-survey-responses.csv")

df.head()
# one way chi-square test

scipy.stats.chisquare(df["Have you ever taken a course in statistics?"].value_counts())

# df["Have you ever taken a course in statistics?"].value_counts() for the respective categorical frequencies
scipy.stats.chisquare(df["Do you have any previous experience with programming?"].value_counts())
# two way chi-square test 

# is there is relation between previous experience with programming and experience in statistics

stats = df["Have you ever taken a course in statistics?"].value_counts()

prog = df["Do you have any previous experience with programming?"].value_counts()

print("\n stats background categories \n",stats)

print("\n programming background categories \n",prog)
#Contingency table

CT = pd.crosstab(df["Do you have any previous experience with programming?"],

                df["Have you ever taken a course in statistics?"])

print(CT)
# two-way chi-square test of independence

scipy.stats.chi2_contingency(CT)