import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd

import calendar

%matplotlib inline

# default figsize for charts

figsize = (12, 6)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
deathRec=pd.read_csv("../input/DeathRecords.csv")

deathBy=pd.read_csv("../input/MannerOfDeath.csv")

deathDay=pd.read_csv("../input/DayOfWeekOfDeath.csv")
print ("\nDeath Records (row x col) ",deathRec.shape)

print ("\nDeath caused (row x col) ",deathBy.shape)

print ("\nDay of death (row x col) ",deathDay.shape)
names=deathRec.columns.values

print ("\n Death Records data headers\n")

print (names)

names=deathBy.columns.values

print ("\n Death caused by headers\n")

print (names)

names=deathDay.columns.values

print ("\n Day of death headers\n")

print (names)
# print the rows with missing data

print ("The count of rows with missing values in Death Records data: \n", deathRec.isnull().sum())

print ("\nThe count of rows with missing values in Death caused by data: \n", deathBy.isnull().sum())

print ("\nThe count of rows with missing values in Death Day data: \n", deathDay.isnull().sum())
# Show the Frequency distribution

deathType=deathRec['MannerOfDeath'].value_counts(sort=True)

deathCnt=deathRec['PlaceOfDeathAndDecedentsStatus'].value_counts(sort=True)



print ("\nDeath Agent code & description\n",deathBy)

print ("\n Death Type\n",deathType)

#print(deathCnt)
# Loading the race data

raceData=pd.read_csv("../input/Race.csv")

print (raceData)
print (deathRec['Race'].value_counts(sort=True))
# load disease description for each Icd10Code

dficd = pd.read_csv('../input/Icd10Code.csv')
# load general data

dffull = pd.read_csv('../input/DeathRecords.csv')
# create a order series with o

# compiled a list of diseases in the number of deaths (from highest to lowest)

deseasesDeathsList = dffull.Icd10Code.value_counts()

deseasesDeathsList.head()
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

s1.head(10)
s2 = pd.Series(icdcodes, index=deseasesDeathsList.index, name='Description')

s2.head(10)


dfRank = pd.concat([s1, s2], axis=1)





dfRank.head(10)
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
alc = list(dficd[dficd['Description'].str.contains('alcohol')]['Code'])

narc = list(dficd[dficd['Description'].str.contains('narcotics')]['Code'])



aloco = dffull[dffull['Icd10Code'].isin(alc)][dffull.Age < 120][dffull.DayOfWeekOfDeath < 8]

narco = dffull[dffull['Icd10Code'].isin(narc)][dffull.Age < 120][dffull.DayOfWeekOfDeath < 8]



plt.title('Deaths by alcohol vs Narcotics', fontsize = 18)

_ = plt.hist(aloco.Age.tolist(),24,alpha=0.9,label='Alcohol')

_ = plt.hist(narco.Age.tolist(),24,alpha=0.4,label='Narcotics')



plt.legend()
suicides = dffull[dffull['MannerOfDeath']==2]

suicides['Age'].hist(bins=range(102))
maritalstatus = pd.read_csv('../input/MaritalStatus.csv')



marital_suicides = suicides.MaritalStatus.value_counts()

print(marital_suicides)

total_marital = dffull.MaritalStatus.value_counts()

print(total_marital)

relative_suicides = marital_suicides/total_marital 

print (relative_suicides)

relative_suicides.plot(kind='bar')
d = pd.merge(dffull,dficd,left_on='Icd10Code',right_on='Code')

ed = pd.read_csv('../input/Education2003Revision.csv')

print(ed['Description'])
d[d['Icd10Code'].str.contains('Y35')]['Education2003Revision'].hist(bins=range(8))