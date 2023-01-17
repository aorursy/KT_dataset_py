# Useful links



# https://www.kaggle.com/cdc/mortality



# http://www.cdc.gov/nchs/data/dvs/Record_Layout_2014.pdf
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import calendar

%matplotlib inline

# default figsize for charts

figsize = (12, 6)
# load disease description for each Icd10Code

dficd = pd.read_csv('../input/Icd10Code.csv')
# load general data

dffull = pd.read_csv('../input/DeathRecords.csv')
# create a order series with o

# compiled a list of diseases in the number of deaths (from highest to lowest)

deseasesDeathsList = dffull.Icd10Code.value_counts()

deseasesDeathsList.head()
# Show table of deseasesDeathsList with Icd10Code description 
# calculate total deaths

totalDeaths = float(sum(deseasesDeathsList.values))

totalDeaths
icdcodes = []

for x in deseasesDeathsList.index:

    tmp = dficd[dficd.Code == x]

    if tmp.empty:

        desc = ''

    else:

        desc = tmp.Description.values[0]

    icdcodes.append(desc)

s1 = pd.Series(deseasesDeathsList.values, index=deseasesDeathsList.index, name='Value')

s2 = pd.Series(icdcodes, index=deseasesDeathsList.index, name='Description')

dfRank = pd.concat([s1, s2], axis=1)



xx = [0]

for x in dfRank.Value / totalDeaths:

    xx.append(xx[-1] + x)

dfRank['AccumRelValue'] = xx[1:]

dfRank['Number'] = range(1,len(xx[1:])+1)

dfRank['Icd10Code'] = dfRank.index



dfRank.head(10)
plt.figure(figsize=figsize)

title = 'Relative value of death '

for n in [9, 99, len(dfRank.AccumRelValue)-1]:

    title += "\n{} diseases claimed the {:.0f}% of lives".format(n+1, dfRank.AccumRelValue[n]*100.)



plt.title(title, fontsize=18)

plt.fill_between(range(dfRank.AccumRelValue.count()), dfRank.AccumRelValue, [0] * dfRank.AccumRelValue.count())

plt.xlabel('count of diseases')

plt.ylabel('relative value of total deaths')

plt.xscale('log')

plt.xlim(0,len(xx))

plt.ylim(0, 1)

plt.legend()
# create new dataframe with part of general data

#

# leaving data when Age less than 120

# leaving data with DayOfWeekOfDeath less than 7

# leaving only 100 of the most dangerous diseases (for simplification)

df = dffull[dffull.Age < 120][dffull.DayOfWeekOfDeath < 8][dffull.Icd10Code.isin(deseasesDeathsList.iloc[0:100].index.tolist())]
plt.figure(figsize=figsize)

plt.title('histogram of death vs age', fontsize=18)

_ = plt.hist(df.Age.tolist(), 20, alpha=0.9, label='M+F')

_ = plt.hist(df[df.Sex == 'M'].Age.tolist(), 20, alpha=0.5, label='M')

_ = plt.hist(df[df.Sex == 'F'].Age.tolist(), 20, alpha=0.5, label='F')

_ = plt.legend()
d = []

for group in sorted(dfRank.Icd10Code.str[0].unique()):

    part = {}

    part['Group'] = group

    part['Value'] = dfRank[dfRank.Icd10Code.str[0] == group]['Value'].sum()

    d.append(part)

    

dfIcdGroup = pd.DataFrame(d).sort_index(by=['Value'])

dfIcdGroup['RelValue'] = dfIcdGroup.Value / totalDeaths



plt.figure(figsize=figsize)

plt.bar([x - 0.4 for x in range(dfIcdGroup.Value.count())], dfIcdGroup.RelValue)

_ = plt.xticks(range(len(dfIcdGroup.Value)), dfIcdGroup.Group)

plt.xlabel('Icd10Code group name')

plt.ylabel('Relative value of deaths')

plt.xlim(0, 23)



title = 'Most Dangerous Icd10Code Groups:'

for x in ['I', 'C', 'J']:

    title += "\n {} - {:.0f}%".format(x, dfIcdGroup[dfIcdGroup.Group == x].RelValue.values[0] * 100.)

plt.title(title, fontsize=18)
DaysInMonths = [calendar.monthrange(2014, x)[1] for x in range(1, 13)]



s = df.MonthOfDeath.value_counts().sort_index()

valuePerDay = s.values / DaysInMonths



plt.figure(figsize=figsize)

plt.title('Absolute value of deaths per day vs month', fontsize=18)

x = np.linspace(0.5, 11.5, 12)

plt.bar(x, valuePerDay)

plt.xticks(range(1, 13), range(1, 13))

plt.ylim(5000, )

plt.xlim(0.5, 12.5)

plt.ylabel('Absolute value of deaths per day')

plt.xlabel('Mount')
plt.figure(figsize=(12, 6))

for st, en in [(40, 60), (60, 80), (80, 100)]:

    s = df[df.Age < en][df.Age > st].MonthOfDeath.value_counts().sort_index()

    valuePerDay = (s.values / DaysInMonths)

    plt.plot(s.index, valuePerDay / valuePerDay.mean(), label="year range {}-{}".format(st, en))



plt.ylabel('Relative value of deaths per day')

plt.xlabel('Month')

plt.xticks(range(1, 13))

plt.xlim(1, 12)

plt.legend()

plt.title("Relative value of death per day for different ages groups", fontsize=18)