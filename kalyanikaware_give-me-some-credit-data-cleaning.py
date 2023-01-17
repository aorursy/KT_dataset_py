# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns
df = pd.read_csv('/kaggle/input/give-me-some-credit-dataset/cs-training.csv')
df.sample(5)
df.rename(columns = {df.columns[0]:'ID'}, inplace = True)
df.info()
df['SeriousDlqin2yrs'].unique()
P = df.groupby(by = 'SeriousDlqin2yrs')['ID'].count().reset_index()
P.rename(columns = {'ID':'Counts'},inplace = True)
P['Percent'] = P['Counts']*100/P['Counts'].sum()
P
# Or simply 

df['SeriousDlqin2yrs'].value_counts(normalize = True).plot(kind= 'barh')
df['RevolvingUtilizationOfUnsecuredLines'].describe()



#it is a percentage column and anything > 1 could be set to NaN
sns.distplot(df['RevolvingUtilizationOfUnsecuredLines'])



# we have an outlier at approx 50000
sns.distplot(df['RevolvingUtilizationOfUnsecuredLines'][df['RevolvingUtilizationOfUnsecuredLines'] <=1])
df['RevolvingUtilizationOfUnsecuredLines'] = df['RevolvingUtilizationOfUnsecuredLines'].apply(lambda x: np.NaN if x >1 else x)
df['RevolvingUtilizationOfUnsecuredLines'].describe()
df['RevolvingUtilizationOfUnsecuredLines'].fillna(df['RevolvingUtilizationOfUnsecuredLines'].mean(),inplace = True)
df['RevolvingUtilizationOfUnsecuredLines'].describe()
df.age.describe()
sns.distplot(df.age)
# since only a few ages are below 18 and above 90, lets cap them



df.loc[df.age > 90, 'age'] = 90

df.loc[df.age < 18, 'age'] = 18
sns.distplot(df.age)
df.drop(columns = ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse'],inplace = True)
df.DebtRatio.describe()



#this is a percentage column and cannot be more than 1.
sns.distplot(df.DebtRatio)
df.DebtRatio[df.DebtRatio > 1].describe()
df.loc[df.DebtRatio > 1, 'DebtRatio'] = np.NaN
df.DebtRatio.fillna(method = 'ffill', inplace = True)

sns.distplot(df.DebtRatio)
df.MonthlyIncome.describe()
sns.distplot(df.MonthlyIncome)
# Count of people with income < 1000

df.ID[df.MonthlyIncome < 1000].count()
#Lets set is as NAN



df.loc[df.MonthlyIncome < 1000, 'MonthlyIncome'] = np.NaN
sns.distplot(df.MonthlyIncome[df.MonthlyIncome < 50000])  # After trying with a few values. I stopped on 5000.
# more than 30000 looks unreasonable according to rest of the data. Hence caping greater incoming

df.loc[df.MonthlyIncome > 30000, 'MonthlyIncome'] = 30000
sns.distplot(df.MonthlyIncome)
df.MonthlyIncome.fillna(method = 'ffill', inplace = True)
df.MonthlyIncome.describe()
sns.distplot(df.MonthlyIncome)
df.NumberOfOpenCreditLinesAndLoans.describe()
sns.distplot(df.NumberOfOpenCreditLinesAndLoans)
# the distribution indicate that it is continuous upto 30. Hence, let's cap at 30



df.loc[df.NumberOfOpenCreditLinesAndLoans > 30, 'NumberOfOpenCreditLinesAndLoans'] = 30
sns.distplot(df.NumberOfOpenCreditLinesAndLoans)
df.NumberRealEstateLoansOrLines.describe()
sns.distplot(df.NumberRealEstateLoansOrLines[df.NumberRealEstateLoansOrLines < 10])
sns.distplot(df.NumberRealEstateLoansOrLines[df.NumberRealEstateLoansOrLines > 10])
# Comparing above 2 graphs, looks like we can cap value to 5.



df.loc[df.NumberRealEstateLoansOrLines > 5, 'NumberRealEstateLoansOrLines'] = 5
sns.distplot(df.NumberRealEstateLoansOrLines)
df.NumberOfDependents.unique()
sns.distplot(df.NumberOfDependents.dropna())
df.NumberOfDependents.describe()
# There are missing values and dependents more than 5 can be capped to max 5.



df.loc[df.NumberOfDependents > 5, 'NumberOfDependents'] = 5



df.NumberOfDependents.describe()
# Since proportion of missing is large, imputation using mean is not appropriate as this will change the distribution too much. 

# we will impute the missing values using ffill as this will preserve the mean and standard deviation.



df.NumberOfDependents.fillna(method = 'ffill', inplace = True)

df.NumberOfDependents.describe()
# Saving the clean file to pickle

pd.to_pickle(df,'CleanCreditFile.pkl')