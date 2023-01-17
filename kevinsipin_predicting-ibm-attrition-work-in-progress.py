import numpy as np

import pandas as pd

import seaborn as sns

import seaborn.matrix as smatrix

import matplotlib.pyplot as plt
df = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')

df.head()
df.isnull().any()
# Plot 1

x = df['Age']

y = df['DailyRate']

sns.jointplot(x, y, kind="kde");



# Plot 2

x = df['Age']

y = df['DistanceFromHome']

sns.jointplot(x, y, kind="kde");



# Plot 3

x = df['Age']

y = df['Education']

sns.jointplot(x, y, kind="kde");



# Plot 4

x = df['JobSatisfaction']

y = df['DailyRate']

sns.jointplot(x, y, kind="kde");



# Plot 5

x = df['YearsAtCompany']

y = df['DailyRate']

sns.jointplot(x, y, kind="kde");



# Plot 6

x = df['Education']

y = df['DailyRate']

sns.jointplot(x, y, kind="kde");
# Select only the numerical variables

list_numerical = ['Age','DailyRate','DistanceFromHome','Education','EmployeeNumber', 

                  'EnvironmentSatisfaction','HourlyRate','JobInvolvement','JobLevel', 

                  'JobSatisfaction','MonthlyIncome','MonthlyRate','NumCompaniesWorked',

                  'PercentSalaryHike','PerformanceRating','RelationshipSatisfaction',

                  'StockOptionLevel','TotalWorkingYears',

                  'TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany',

                  'YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']



data = df[list_numerical] # We can use data.head() to check if everything went correct



fig, ax = plt.subplots()



# Meh, doens't work out yet. Try on the whiteboard first

sns.heatmap(data, cbar=False, squara=False,

            robust=True, annot=True, fmt=".1d",

            annot_kws={"size":8}, linewidths=0.5, 

            cmap="RdYlGn", ax=ax)