# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
survey=pd.read_csv('../input/survey_results_public.csv')
survey.shape
survey.head()
print('The Total Responses Recorded are:',survey.shape[0])
plt.subplots(figsize=(12,6))

ax=survey['Country'].value_counts()[:10].plot.bar(width=0.8,color='#f45c42')

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.10, p.get_height()+0.25))

plt.xlabel('Country')

plt.ylabel('Count')

plt.show()
plt.subplots(figsize=(12,6))

students=survey[survey['Professional']=='Student']

ax=students['Country'].value_counts()[:10].plot.bar(width=0.8,color='#f45c42')

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.10, p.get_height()+0.25))

plt.xlabel('Country')

plt.ylabel('Count')

plt.show()
l1=list(students.columns)

for i in ['C','C++','C#','Java','Python','R','JavaScript','PHP']:

    print(i,':',survey['HaveWorkedLanguage'].apply(lambda x: i in str(x).split('; ')).value_counts()[1])
d={}

for i in ['C','C++','C#','Java','Python','R']:

    d[i]=(survey['HaveWorkedLanguage'].apply(lambda x: i in str(x).split('; ')).value_counts()[1])

    

lang=pd.DataFrame(list(d.items()))

lang.columns=[['Language','Count']]

lang.set_index('Language',inplace=True)    

lang.plot.barh(width=0.8,color='#005544')

plt.show()
stu=students.groupby(['Country','HaveWorkedLanguage'])['Respondent'].count().reset_index()

stu.columns=[['Country','Known_Languages','Count']]

stu=stu.Known_Languages.str.get_dummies('; ').groupby(stu.Country).sum()

stu=stu.idxmax(1).reset_index()

stu.columns=[['Country','Language']]

stu[stu['Country'].isin(survey['Country'].value_counts()[:10].index)]
stu=stu['Language'].value_counts().reset_index()

stu.set_index('index',inplace=True)

stu.plot(kind='barh',width=0.8,color='#a454ff')

plt.show()