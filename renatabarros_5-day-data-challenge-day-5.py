import numpy as np

import pandas as pd

import scipy.stats

import matplotlib.pyplot as plt



# Read the file into a dataset

data = pd.read_csv("../input/database.csv")



# See first few rows

data.head()

# Pick two columns with categorical variables in it = operator and airport

operators = data["Operator"]

aircrafts = data["Aircraft Type"]



# Start with one way chi-square test (one column)

print("Chi-square test applied to operators")

print(scipy.stats.chisquare(operators.value_counts()))



# One way chi-square test (other column)

print("Chi-square test applied to aircrafts")

print(scipy.stats.chisquare(aircrafts.value_counts()))
# For a two way chi-square test, a contingency table is needed

# Contingency table: matrix that displays the multivariate frequency distribution of variables.



contingencyTable = pd.crosstab(operators, aircrafts)



scipy.stats.chi2_contingency(contingencyTable)