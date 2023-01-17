import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import calendar

%matplotlib inline
# load disease description for each Icd10Code

dficd = pd.read_csv('../input/Icd10Code.csv')



dataset =  pd.read_csv('../input/DeathRecords.csv', header=0)

print(len(dataset))
mannerofdeath = pd.read_csv('../input/MannerOfDeath.csv')
print(mannerofdeath)
#cleaning data with bad age death, bad days of the week and not natural deaths.

df = dataset[dataset.Age < 120][dataset.DayOfWeekOfDeath < 8][~dataset.MannerOfDeath.isin([1,2,3,6])]

print(len(df))
# number of deaths by sex

print('men deaths in the dataset:', len(df[df.Sex == 'M']))

print('women deaths in the dataset:', len(df[df.Sex == 'F']))
print (df[df.Sex == 'M'].Age.mean())

print (df[df.Sex == 'F'].Age.mean())
plt.figure()

plt.title('histogram of death vs age by sex', fontsize=18)

_ = plt.hist(df[df.Sex == 'M'].Age.tolist(), 10, alpha=0.5, label='M')

_ = plt.hist(df[df.Sex == 'F'].Age.tolist(), 10, alpha=0.5, label='F')

_ = plt.legend()
malesDeaths  = df[df.Sex == 'M']

femaleDeaths = df[df.Sex == 'F']



for i in range(0, 10):

    print("age between ",i*10,(i+1)*10,"difference in deaths (men deaths - women deaths):", len(malesDeaths[np.logical_and(malesDeaths.Age>i*10,malesDeaths.Age<(i+1)*10)])-len(femaleDeaths[np.logical_and(femaleDeaths.Age>i*10,femaleDeaths.Age<(i+1)*10)]))
malesDeaths60s=malesDeaths[malesDeaths.Age>=60][malesDeaths.Age<70]

dficd.columns= ['Icd10Code', 'dx']

malesDeaths60s =  pd.merge(malesDeaths60s, dficd, how='left', on='Icd10Code')
print ("number of men deaths in 60s",len(malesDeaths60s))
men60counts = malesDeaths60s[['Icd10Code', 'Id']].groupby(['Icd10Code'], as_index=False).count()
most_common_causes = pd.merge(men60counts, dficd, how='left', on='Icd10Code')

most_common_causes = most_common_causes.sort_values(by=['Id'],ascending=False)

most_common_causes.head(15)
femaleDeaths60s=femaleDeaths[femaleDeaths.Age>=60][femaleDeaths.Age<70]

dficd.columns= ['Icd10Code', 'dx']

femaleDeaths60s =  pd.merge(femaleDeaths60s, dficd, how='left', on='Icd10Code')
print ("number of women deaths in 60s",len(femaleDeaths60s))
women60counts = femaleDeaths60s[['Icd10Code', 'Id']].groupby(['Icd10Code'], as_index=False).count()
most_common_causes = pd.merge(women60counts, dficd, how='left', on='Icd10Code')

most_common_causes = most_common_causes.sort_values(by=['Id'],ascending=False)

most_common_causes.head(15)