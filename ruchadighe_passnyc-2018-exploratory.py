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
school_data= pd.read_csv('../input/2016 School Explorer_new.csv')
def num_missing(x):
  return sum(x.isnull())

#print(school_data.apply(num_missing, axis=0))
plt.hist(school_data.Latitude)
plt.title("Histogram of Latitute")
plt.xlabel("Latitute")
plt.ylabel("Frequency")
plt.show()
tab_City=pd.crosstab(index=school_data['City'], columns='count', colnames=[''])
#print(tab_City)
tab_City_School=pd.crosstab(school_data['City'],school_data['Community School?'],margins=True)
ax = sns.heatmap(tab_City_School, annot=True, fmt="d", cmap="YlGnBu")
a=school_data[~np.isnan(school_data['School Income Estimate'])]
plt.boxplot(a['School Income Estimate'])
a.boxplot(column="School Income Estimate",by="Community School?")
a.hist(column="School Income Estimate",by="Community School?",bins=30)


group_school = school_data.groupby('Community School?')['Percent Asian']
group_school.plot(kind='hist', bins=30, figsize=[12,6], alpha=.4, legend=True)
school_data.hist(column="Percent Asian",by="Community School?",bins=30)
target_0 = school_data.loc[school_data['Community School?'] == 'Yes']
target_1 = school_data.loc[school_data['Community School?'] == 'No']
sns.distplot(target_0[['Percent Asian']], hist=False, rug=True)
sns.distplot(target_1[['Percent Asian']], hist=False, rug=True)
group_student_absentism = school_data.groupby('Supportive Environment Rating')['Percent of Students Chronically Absent']
group_student_absentism.plot(kind='hist', bins=30, figsize=[12,6], alpha=.4, legend=True)
school_data.hist(column="Percent of Students Chronically Absent",by="Supportive Environment Rating",bins=30)
df_temp=school_data[school_data['Supportive Environment Rating'].notnull()]
df_dummies = pd.get_dummies(df_temp['Supportive Environment Rating'])
del df_dummies[df_dummies.columns[-1]]
df_new = pd.concat([df_temp['Percent of Students Chronically Absent'], df_dummies], axis=1)
x = df_new.values
correlation_matrix = np.corrcoef(x.T)
print(correlation_matrix)
