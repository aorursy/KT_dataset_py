# import our libraries
import scipy.stats # statistics
import pandas as pd # dataframe

# read in our data
surveyData = pd.read_csv("../input/5day-data-challenge-signup-survey-responses/anonymous-survey-responses.csv")
surveyData["Have you ever taken a course in statistics?"].value_counts()
digimon = pd.read_csv("../input/digidb/DigiDB_digimonlist.csv")
digimon
digimon["Attribute"].value_counts()
digimon["Stage"].value_counts()
# first let's do a one-way chi-squared test for stats background
scipy.stats.chisquare(surveyData["Have you ever taken a course in statistics?"].value_counts())
print(scipy.stats.chisquare.__doc__)
scipy.stats.chisquare(digimon["Attribute"].value_counts())
digimon_two_categorical_values = pd.crosstab(digimon["Attribute"], digimon["Stage"])
scipy.stats.chi2_contingency(digimon_two_categorical_values)
surveyData["Do you have any previous experience with programming?"].value_counts()
# first let's do a one-way chi-squared test for programming background
scipy.stats.chisquare(surveyData["Do you have any previous experience with programming?"].value_counts())

# now let's do a two-way chi-square test. Is there a relationship between programming background 
# and stats background?

contingencyTable = pd.crosstab(surveyData["Do you have any previous experience with programming?"],
                              surveyData["Have you ever taken a course in statistics?"])

contingencyTable
scipy.stats.chi2_contingency(contingencyTable)