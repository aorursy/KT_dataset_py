import scipy.stats # statistics

import pandas as pd # dataframe



# read the data

survey_data = pd.read_csv("../input/survey_results_public.csv")

survey_data.head()
scipy.stats.chisquare(survey_data["ProgramHobby"].value_counts())
scipy.stats.chisquare(survey_data["University"].value_counts())
contingencyTable = pd.crosstab(survey_data["University"], survey_data["ProgramHobby"])
contingencyTable
scipy.stats.chi2_contingency(contingencyTable)