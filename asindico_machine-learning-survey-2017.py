# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
mc = pd.read_csv('../input/multipleChoiceResponses.csv',encoding='ISO-8859-1')

mc.head()
f,axa = plt.subplots(1,1,figsize=(15,6))

tmp = mc.groupby(['GenderSelect'],as_index=True)["GenderSelect"].count().reset_index(name="count")

sns.barplot(x=tmp['GenderSelect'],y=tmp['count'])

axa.set_title("Genders Histogram")

plt.show()
f,axa = plt.subplots(1,1,figsize=(15,6))

sns.distplot(mc[mc['Age'].isnull()==False]['Age'])

axa.set_title('Age Distribution')

plt.show()
f,axa = plt.subplots(1,1,figsize=(15,10))

tmp = mc.groupby(['EmploymentStatus'],as_index=True)["EmploymentStatus"].count().reset_index(name="count")

sns.barplot(y=tmp['EmploymentStatus'],x=tmp['count'],orient='h')

axa.set_title("EmploymentStatus Histogram")

plt.show()
mc['EmploymentStatus'].unique()
#JobSkillImportanceKaggleRanking

f,axa = plt.subplots(2,1,figsize=(15,10))



sel = mc[mc['EmploymentStatus']=='Independent contractor, freelancer, or self-employed']

sel = sel[sel['JobSkillImportanceKaggleRanking'].isnull()==False]

sel = sel.groupby(['JobSkillImportanceKaggleRanking'],as_index=True)["JobSkillImportanceKaggleRanking"].count().reset_index(name="count")

sns.barplot(y=sel['JobSkillImportanceKaggleRanking'],x=sel['count'],ax=axa[0],orient='h')

axa[0].set_title('Perceived Kaggle Ranking Importance in Freelancers')



sel = mc[mc['EmploymentStatus']=='Employed full-time']

sel = sel[sel['JobSkillImportanceKaggleRanking'].isnull()==False]

sel = sel.groupby(['JobSkillImportanceKaggleRanking'],as_index=True)["JobSkillImportanceKaggleRanking"].count().reset_index(name="count")

sns.barplot(y=sel['JobSkillImportanceKaggleRanking'],x=sel['count'],ax=axa[1],orient='h')

axa[1].set_title('Perceived Kaggle Ranking Importance in Full Time Employees')



plt.show()