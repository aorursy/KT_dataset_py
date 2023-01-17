# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')

data.info()

print(data)
data.describe()
data.corr()
f,ax = plt.subplots(figsize=(5, 5))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()

data.columns
s=data['math score']

a=s.sort_values()

t=0

for i in a:

    b=s.max()

    if i== b:

        t+=1

print(a)

print(b)

print(t)

data.head()
data = data.rename(columns={'gender':'gender', 'race/ethnicity':'race/ethnicity','parental level of education': 'parental_level_of_education', 

                        'test preparation course': 'test_preparation_course','math score':'math_score',

                       'reading score':'reading_score','writing score':'writing_score'})

# Or rename the existing DataFrame (rather than creating a copy) 

data.rename(columns={'gender':'gender', 'race/ethnicity':'race/ethnicity','parental level of education': 'parental_level_of_education', 

                        'test preparation course': 'test_preparation_course','math score':'math_score',

                       'reading score':'reading_score','writing score':'writing_score'}, inplace=True)

data.math_score.plot(kind = 'line', color = 'g',label = 'math_score',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.reading_score.plot(color = 'r',label = 'reading_score',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()

data.plot(kind='scatter', x='writing_score', y='math_score',alpha = 0.5,color = 'red')

plt.xlabel('writing_score')              # label = name of label

plt.ylabel('math_score')

plt.title('writing_score math_score Scatter Plot')            # title = title of plot

plt.show()
data.math_score.plot(kind = 'hist',bins = 70,figsize = (12,12))

plt.show()
a=data[(data['gender']!='female') & (data['math_score']>85) & (data['writing_score']>85) & (data['reading_score']>85)]

print(a)

e=a.count()

print(e)

b=data[(data['gender']!='male') & (data['math_score']>85) & (data['writing_score']>85) & (data['reading_score']>85)]

k=b.count()

print(k)
list1 = list(data['math_score'])

list2 = list(data['gender'])

z = zip(list1,list2)

print(z)

z_list = list(z)

print(z_list)
female_m_score =[]

for i in range(len(z_list)):

    if z_list[i][1]=='female':

        female_m_score.append(z_list[i][0])

#female_m_score =[z_list[i][0] if z_list[i][1]=='female' for i in range(len(z_list)) ]

#print(female_m_score)

#print(female_m_score.sort())

print(len(female_m_score))

max=female_m_score[0]

for i in range(len(female_m_score)):

    female_m_score[i]>max

    max,female_m_score[i]=female_m_score[i],max

for i in range(len(female_m_score)):

    print(female_m_score[i],end=' ')

threshold=sum(data.math_score)/len(data.math_score)

data["math_score_level"]=["successful" if i>threshold else "unsuccessful" for i in data.math_score]

data.loc[:10,["math_score_level","math_score"]]
print(data)
print(data['gender'].value_counts(dropna=False))
data.dropna(inplace=True)

data.describe()
data.boxplot(column='math_score',by = 'gender')
melted = pd.melt(frame=data, id_vars = 'race/ethnicity', value_vars = ['test_preparation_course','lunch'])

melted
#melted.pivot(index = 'race/ethnicity',columns = 'variable', values = 'value')
data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row