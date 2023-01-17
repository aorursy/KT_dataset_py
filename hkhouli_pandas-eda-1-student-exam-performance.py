# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
stuperf = pd.read_csv('../input/StudentsPerformance.csv')
print(stuperf.shape)
print(stuperf.dtypes)
stuperf.head(10)
stuperf = stuperf.rename(columns={'race/ethnicity':'race', 'test preparation course' : 'tpc'})
stuperf.tpc.replace('none', 'no', inplace=True)
stuperf.tpc.replace('completed', 'yes', inplace=True)

stuperf.head(5)

stuperf.isnull().sum()
stuperf.describe()
print("Gender Distribution")
print(stuperf.gender.value_counts(normalize=True))

print('Race Distribution')
print(stuperf.race.value_counts(normalize=True))
stuperf.race.value_counts().plot.bar()
stuperf.groupby('gender').race.value_counts().plot.bar()
stuperf['overall score'] = pd.Series(stuperf['math score'] + stuperf['reading score'] + stuperf['writing score'], index=stuperf.index)
stuperf.head()
stuperf.boxplot(column='overall score', grid=False)
corrs = stuperf.corr()
fig, ax = plt.subplots(figsize=(8,8))
dropself = np.zeros_like(corrs)
dropself[np.triu_indices_from(dropself)] = True
sns.heatmap(corrs, cmap='PuRd', annot=True, fmt='.2f', mask=dropself)
plt.xticks(range(len(corrs.columns)), corrs.columns)
plt.yticks(range(len(corrs.columns)), corrs.columns)
plt.show()
plt.plot(stuperf['writing score'], stuperf['reading score'], 'b.')
plt.xlabel('Writing Score')
plt.ylabel('Reading Score')
plt.grid(False)
plt.show()
wscores = stuperf['writing score']
rscores = stuperf['reading score']
xvals = np.linspace(0,100,100)
coeffs = np.polyfit(wscores, rscores, 1)
poly = np.poly1d(coeffs)
plt.plot(xvals, poly(xvals), 'r-')
plt.xlabel('Writing Score')
plt.ylabel('Reading Score')
plt.title('Fitted line of writing score and corresponding expected reading score')
plt.show()
a_stu = stuperf.loc[stuperf['overall score']/300 >= 0.9]
b_stu = stuperf.loc[(stuperf['overall score']/300 >= 0.8) & (stuperf['overall score']/300 < 0.9) ]
c_stu = stuperf.loc[(stuperf['overall score']/300 >= 0.7) & (stuperf['overall score']/300 < 0.8) ]
d_stu = stuperf.loc[(stuperf['overall score']/300 >= 0.6) & (stuperf['overall score']/300 < 0.7) ]
f_stu = stuperf.loc[stuperf['overall score']/300 < 0.6 ]
fig, axarr = plt.subplots(3,2, figsize=(15,10))
fig.tight_layout()
a_stu.boxplot(by='parental level of education', column='overall score', ax=axarr[0][0], fontsize=7, grid=False)
b_stu.boxplot(by='parental level of education', column='overall score', ax=axarr[0][1], fontsize=7, grid=False)
c_stu.boxplot(by='parental level of education', column='overall score', ax=axarr[1][0], fontsize=7, grid=False)
d_stu.boxplot(by='parental level of education', column='overall score', ax=axarr[1][1], fontsize=7, grid=False)
f_stu.boxplot(by='parental level of education', column='overall score', ax=axarr[2][0], fontsize=7, grid=False)
fig, axarr = plt.subplots(3,2, figsize=(20,10))
fig.tight_layout()
sns.violinplot(x='parental level of education', y='overall score',data=a_stu.sort_values(by='parental level of education'), ax=axarr[0][0])
sns.violinplot(x='parental level of education', y='overall score',data=b_stu.sort_values(by='parental level of education'), ax=axarr[0][1])
sns.violinplot(x='parental level of education', y='overall score',data=c_stu.sort_values(by='parental level of education'), ax=axarr[1][0])
sns.violinplot(x='parental level of education', y='overall score',data=d_stu.sort_values(by='parental level of education'), ax=axarr[1][1])
sns.violinplot(x='parental level of education', y='overall score',data=f_stu.sort_values(by='parental level of education'), ax=axarr[2][0])
sns.violinplot(x='parental level of education', y='overall score', data=stuperf.sort_values(by='parental level of education'), figsize=(15,10), rotation=90)
stuperf.groupby('parental level of education')['overall score'].std()
pled_ind = stuperf.sort_values(by='parental level of education')
pled = pd.DataFrame({'Mean Score':list(stuperf.groupby('parental level of education')['overall score'].mean()),
                    'Standard Deviation': list(stuperf.groupby('parental level of education')['overall score'].std())},
                   index=pled_ind['parental level of education'].unique())
pled
hs_ed = stuperf.loc[(stuperf['parental level of education'] == 'some high school') |
                    (stuperf['parental level of education'] == 'high school')]
hs_ed.head()
hs_ed.groupby(['tpc'])['parental level of education'].value_counts().plot.bar(stacked=True)
hs_ed.groupby(['parental level of education']).tpc.value_counts()
prep = stuperf.loc[stuperf.tpc == 'yes']
noprep = stuperf.loc[stuperf.tpc == 'no']
print('Scores w/ completion of tpc')
print(prep['overall score'].describe())
print('\nScores w/out completion of tpc')
print(noprep['overall score'].describe())