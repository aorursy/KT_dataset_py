import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



data = pd.read_csv('../input/survey_results_public.csv')
data['Salary'].unique()
data = data[data['Salary'].notnull()]

np.sort(data['Salary'].unique())
data['Country'].unique()
data = data.query('Salary>21')

salary = data['Salary'].unique()

print('Salary range (USD): ' + str(min(salary)) + ' - ' + str(max(salary)))
data['Salary'].describe()
data.Salary.plot.hist(figsize=(15,10))
# This code is taken and modified from 

# https://www.kaggle.com/m2skills/simple-exploratory-analysis-visualizations-more



# Get list of unique developer types

developerType = set()

salary_avg_jobs = data['DeveloperType'].drop(data.loc[data['DeveloperType'].isnull()].index)

for datum in salary_avg_jobs:

    for types in datum.split(';'):

        developerType.add(types.strip())

developerType = sorted(list(developerType))



# Clean data, make sure no DeveloperType is null

salary_avg_jobs = data[data['DeveloperType'].notnull()]



# Prepare developer type index to orgainze the salary

devDict = {}

for index, dev in enumerate(developerType):

    devDict[dev] = index



# Organize the salary based on its job type

devSalaries = [[] for i in range(len(developerType))]

for index, datum in salary_avg_jobs.iterrows():

    devlist = datum['DeveloperType'].split(";")

    for d in devlist:

        devSalaries[devDict[d.strip()]].append(datum['Salary'])



# Calculate the average salary for each job type

Salaries = []

for sal in devSalaries:

    Salaries.append(np.mean(sal))



# Construct the data frame

devSalaries = pd.DataFrame()

devSalaries["DeveloperType"] = developerType

devSalaries["AverageSalary"] = Salaries



# Plot

plt.subplots(figsize=(15,7))

sns.set_style("whitegrid")

sal = sns.barplot(x=devSalaries.DeveloperType, y=devSalaries.AverageSalary, orient = 1)

sal.set_xticklabels(devSalaries.DeveloperType, rotation=90)
from functools import reduce



# Compile list of language found in survey

language = map((lambda x: str(x).split('; ')), 

               data['HaveWorkedLanguage'])

# Flatten the list

language = reduce((lambda x, y: x + y), language)

# Remove duplication

language = list(set(language))



# Count the language users

languageUser={}

for i in language:

    languageUser[i] = data['HaveWorkedLanguage'].apply(

        lambda x: i in str(x).split('; ')).value_counts()[1]



# Start plotting

lang = pd.DataFrame(list(languageUser.items()))

lang.columns = [['Language', 'Count']]

lang.set_index('Language', inplace=True)

lang.sort_values('Count', inplace=True)

lang.plot.barh(width=0.8, color='#005544', figsize=(15,25))

plt.show()
# Clean data, make sure no HaveWorkedLanguage is null

salary_avg_lang = data[data['HaveWorkedLanguage'].notnull()]



# Prepare developer type index to orgainze the salary

devDict = {}

for index, dev in enumerate(language):

    devDict[dev] = index



# Organize the salary based on its job type

devSalaries = [[] for i in range(len(language))]

for index, datum in salary_avg_lang.iterrows():

    devlist = datum['HaveWorkedLanguage'].split("; ")

    if not devlist:

        continue

    for d in devlist:

        devSalaries[devDict[d.strip()]].append(datum['Salary'])

        

# Calculate the average salary for each job type

Salaries = list(map(lambda x: np.mean(x), devSalaries))



# Construct the data frame

lang = pd.DataFrame()

lang["Language"] = language

lang["AverageSalary"] = Salaries

lang.columns = [['Language', 'Salary']]

lang.set_index('Language', inplace=True)

lang.sort_values('Salary', inplace=True)

lang.plot.barh(width=0.8, color='#005544', figsize=(15,25))

plt.show()