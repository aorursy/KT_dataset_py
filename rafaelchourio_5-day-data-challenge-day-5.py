# Import Required Libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Check for dataset availability

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

print(check_output(["ls", "../input/80-cereals"]).decode("utf8"))
# Load First Dataset and view the first registers

df1=pd.read_csv('../input/80-cereals/cereal.csv')

df1.head(2)
print(check_output(["ls", "../input/5day-data-challenge-signup-survey-responses/"]).decode("utf8"))
print(check_output(["ls", "../input/5day-data-challenge-signup-survey-responses/anonymous-survey-responses.csv"]).decode("utf8"))
# Load Second Dataset and view the first registers

df2=pd.read_csv('../input/5day-data-challenge-signup-survey-responses/anonymous-survey-responses.csv')

df2.head(2)
mfr=df1['mfr'].value_counts()

mfr.head()
type=df1['type'].value_counts()

type.head()
chisquare(mfr)
chisquare(type)
contigencyTable1=pd.crosstab(df1['mfr'],df1['type'])

contigencyTable1
scipy.stats.chi2_contingency(contigencyTable1)
fig,(ax1,ax2)=plt.subplots(ncols=2, figsize=(12,6))

sns.countplot(x='mfr',data=df1,ax=ax1)

sns.countplot(x='type',data=df1,ax=ax2)
sns.pairplot(df1)
scipy.stats.chisquare(df2['Have you ever taken a course in statistics?'].value_counts())
scipy.stats.chisquare(df2['Do you have any previous experience with programming?'].value_counts())
contingencyTable2=pd.crosstab(df2['Do you have any previous experience with programming?'],df2['Have you ever taken a course in statistics?'])
dd
contingencyTable2
scipy.stats.chi2_contingency(contingencyTable2)