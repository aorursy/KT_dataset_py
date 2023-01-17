import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



crime = pd.read_csv("../input/crime.csv")

crime
sns.lmplot('Population', 'Murder rate', data=crime, fit_reg=False)
sns.distplot(crime[['Population']])
sns.distplot(crime[['Murder rate']])
crime.mean()
crime.median()