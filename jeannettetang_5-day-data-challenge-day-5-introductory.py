import pandas as pd



data = pd.read_csv("../input/Health_AnimalBites.csv")
import scipy.stats



# one-way chi-squared test for Species. 

# stats = the chi-squared value, where larger means more difference from a uniform distribution

scipy.stats.chisquare(data["SpeciesIDDesc"].value_counts())
# one-way chi-squared test for Where Bitten

# stats = the chi-squared value, where larger means more difference from a uniform distribution

scipy.stats.chisquare(data["WhereBittenIDDesc"].value_counts())
# two-way chi-squared test to see if there is a relationship between Species and where they are bitten

contingency_table = pd.crosstab(data["SpeciesIDDesc"], data["WhereBittenIDDesc"])



# (chi-square, p-value, degrees of freedom, ...)

# larger chi-square means stronger relationship

chi = scipy.stats.chi2_contingency(contingency_table)