# Import modules

import pandas as pd

import numpy as np



# Read in dataset

df = pd.read_csv('../input/bank_data.csv')



# Print the first few rows of the DataFrame

df.head()
# Scatter plot of Age vs. Expected Recovery Amount

from matplotlib import pyplot as plt

%matplotlib inline

plt.scatter(x=df['expected_recovery_amount'], y=df['age'], c="g", s=2)

plt.xlim(0, 2000)

plt.ylim(0, 60)

plt.xlabel("Expected Recovery Amount")

plt.ylabel("Age")

plt.legend(loc=2)

plt.show()
# Import stats module

from scipy import stats



# Compute average age just below and above the threshold

era_900_1100 = df.loc[(df['expected_recovery_amount']<1100) & 

                      (df['expected_recovery_amount']>=900)]

by_recovery_strategy = era_900_1100.groupby(['recovery_strategy'])

by_recovery_strategy['age'].describe().unstack()



# Perform Kruskal-Wallis test 

Level_0_age = era_900_1100.loc[df["recovery_strategy"]=="Level 0 Recovery"]['age']

Level_1_age = era_900_1100.loc[df["recovery_strategy"]=="Level 1 Recovery"]['age']

stats.kruskal(Level_0_age,Level_1_age) 
# Number of customers in each category

crosstab = pd.crosstab(df.loc[(df['expected_recovery_amount']<1100) & 

                              (df['expected_recovery_amount']>=900)]['recovery_strategy'], 

                       df['sex'])

print(crosstab)



# Chi-square test

chi2_stat, p_val, dof, ex = stats.chi2_contingency(crosstab)

print(p_val)
# Scatter plot of Actual Recovery Amount vs. Expected Recovery Amount 

plt.scatter(x=df['expected_recovery_amount'], 

            y=df['actual_recovery_amount'], c="g", s=2)

plt.xlim(900, 1100)

plt.ylim(0, 2000)

plt.xlabel("Expected Recovery Amount")

plt.ylabel("Actual Recovery Amount")

plt.legend(loc=2)

plt.show()
# Compute average actual recovery amount just below and above the threshold

by_recovery_strategy['actual_recovery_amount'].describe().unstack()



# Perform Kruskal-Wallis test

Level_0_actual = era_900_1100.loc[df['recovery_strategy']=='Level 0 Recovery']['actual_recovery_amount']

Level_1_actual = era_900_1100.loc[df['recovery_strategy']=='Level 1 Recovery']['actual_recovery_amount']

stats.kruskal(Level_0_actual,Level_1_actual) 



# Repeat for a smaller range of $950 to $1050

era_950_1050 = df.loc[(df['expected_recovery_amount']<1050) & 

                      (df['expected_recovery_amount']>=950)]

Level_0_actual = era_950_1050.loc[df['recovery_strategy']=='Level 0 Recovery']['actual_recovery_amount']

Level_1_actual = era_950_1050.loc[df['recovery_strategy']=='Level 1 Recovery']['actual_recovery_amount']

stats.kruskal(Level_0_actual,Level_1_actual)
# Import statsmodels

import statsmodels.api as sm



# Define X and y

X = era_900_1100['expected_recovery_amount']

y = era_900_1100['actual_recovery_amount']

X = sm.add_constant(X)



# Build linear regression model

model = sm.OLS(y, X).fit()

predictions = model.predict(X)



# Print out the model summary statistics

model.summary()
# Create indicator (0 or 1) for expected recovery amount >= $1000

df['indicator_1000'] = np.where(df['expected_recovery_amount']<1000, 0, 1)

era_900_1100 = df.loc[(df['expected_recovery_amount']<1100) & 

                      (df['expected_recovery_amount']>=900)]



# Define X and y

X = era_900_1100[['expected_recovery_amount', 'indicator_1000']]

y = era_900_1100['actual_recovery_amount']

X = sm.add_constant(X)



# Build linear regression model

model = sm.OLS(y,X).fit()



# Print the model summary

model.summary()
# Redefine era_950_1050 so the indicator variable is included



era_950_1050 = df.loc[(df['expected_recovery_amount']<1050) & 

                      (df['expected_recovery_amount']>=900)]





# Define X and y 

X = era_950_1050[['expected_recovery_amount','indicator_1000']]

y = era_950_1050['actual_recovery_amount']

X = sm.add_constant(X)



# Build linear regression model

model = sm.OLS(y,X).fit()



# Print the model summary

model.summary()