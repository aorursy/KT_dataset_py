# I'm interested in finding out if there's a relationship between having programming background and having taken statistics
# Now let's do a chi-square test! The chisquare function from scipy.stats will only do a one-way comparison, so let's start with that.

# import our libraries
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plotting
import scipy.stats # statistics

# read in our data
survey = pd.read_csv("../input/anonymous-survey-responses.csv")

scipy.stats.chisquare(survey["Have you ever taken a course in statistics?"].value_counts())
# first let's do a one-way chi-squared test for programming background
scipy.stats.chisquare(survey["Do you have any previous experience with programming?"].value_counts())
# now let's do a two-way chi-square test. Is there a relationship between programming background 
# and stats background?

contingencyTable = pd.crosstab(survey["Do you have any previous experience with programming?"],
                              survey["Have you ever taken a course in statistics?"])

scipy.stats.chi2_contingency(contingencyTable)