# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import calendar

%matplotlib inline
dficd = pd.read_csv('../input/Icd10Code.csv')



dataset =  pd.read_csv('../input/DeathRecords.csv', header=0)

print(len(dataset))
mannerofdeath = pd.read_csv('../input/MannerOfDeath.csv')
print(mannerofdeath)
print(mannerofdeath)
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
femaleDeaths60s=femaleDeaths[femaleDeaths.Age>=60][femaleDeaths.Age<70]

dficd.columns= ['Icd10Code', 'dx']

femaleDeaths60s =  pd.merge(femaleDeaths60s, dficd, how='left', on='Icd10Code')
print ("number of women deaths in 60s",len(femaleDeaths60s))