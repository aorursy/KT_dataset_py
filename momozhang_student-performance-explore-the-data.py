import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sp=pd.read_csv('../input/StudentsPerformance.csv')
## Explore the data
sp.head()
sp.shape[0]
## Explore each column
sp.columns.unique()
print(sp['lunch'].unique())
print(sp['race/ethnicity'].unique())
print(sp['parental level of education'].unique())
print(sp['test preparation course'].unique())
### Explore data using pivot table
sp.pivot_table(index='parental level of education',columns=['lunch','race/ethnicity'],values='math score',aggfunc=np.mean)

### WE can get: lunch,ethic group and parental level of education may all have influence on education.
### Explore scores data
mavg=sp['math score'].mean()
ravg=sp['reading score'].mean()
wavg=sp['writing score'].mean()
avg=np.array([mavg,ravg,wavg])
avg
mstd=sp['math score'].std()
rstd=sp['reading score'].std()
wstd=sp['writing score'].std()
std=np.array([mstd,rstd,wstd])
std
plt.hist(sp['math score'])
plt.show()
plt.hist(sp['reading score'])
plt.show()
plt.hist(sp['writing score'])
plt.show()
## The distribution,mean and standard deviation of three scores are similar. 
## We will calculate the mean of three scores for each student and use this avg_score to represent student perfprmance 
avg_score=sp['avg_score'] = sp[['math score','reading score','writing score']].mean(axis=1)
sp.head()
race_score=sp.groupby('race/ethnicity')['avg_score'].mean()
paedu_score=sp.groupby('parental level of education')['avg_score'].mean()
lunch_score=sp.groupby('lunch')['avg_score'].mean()
testpre_score=sp.groupby('test preparation course')['avg_score'].mean()
layout=(4,1,1)
plt.plot(race_score)
plt.title('race score')
plt.xlabel('race')
plt.ylabel('avg_score')
plt.show()
layout=(4,1,2)
plt.plot(paedu_score)
plt.title('paedu score')
plt.xlabel('parental edu')
plt.ylabel('avg_score')
plt.xticks(rotation=60)
plt.show()
layout=(4,1,3)
plt.plot(lunch_score)
plt.title('lunch score')
plt.xlabel('lunch')
plt.ylabel('avg_score')
plt.show()
layout=(4,1,4)
plt.plot(testpre_score)
plt.title('test preparation')
plt.xlabel('test preparation')
plt.ylabel('avg_score')
plt.show()
