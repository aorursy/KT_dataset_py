# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt #data visualization

import seaborn as sns #data visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/StudentsPerformance.csv')
#Adding average of all scores to dataset

data['avg'] = (data['math score']+data['reading score'] + data['writing score'])/3
#Distributions of scores

plt.figure(figsize=(15,10))

sns.kdeplot(data['math score'], label ='math')

sns.kdeplot(data['reading score'], label ='reading')

sns.kdeplot(data['writing score'], label ='writing')

plt.title('KDE plots of exam scores')
#Distribution of math score

plt.figure(figsize=(15,10))

sns.distplot(data['math score'], label='math')
#Distribution of reading score

plt.figure(figsize=(15,10))

sns.distplot(data['reading score'], label='reading')
#Distribution of writing score

plt.figure(figsize=(15,10))

sns.distplot(data['writing score'], label='writing')
data.head()
#Distribution of math score for different genders

sns.catplot(x='gender', y='math score', kind='boxen', data=data, height=8, aspect=1)
#Distribution of reading score for different genders

sns.catplot(x='gender', y='reading score', kind='boxen', data=data, height=8, aspect=1)
#Distribution of writing score for different genders

sns.catplot(x='gender', y='writing score', kind='boxen', data=data, height=8, aspect=1)
#Distribution of average score for 3 exams for different genders

sns.catplot(x='gender', y='avg', kind='boxen', data=data, height=8, aspect=1)
data.head()
#Influence of preparation on average score

sns.catplot(x='test preparation course', y='avg', data=data, kind='boxen', height=8, aspect=1)
#Influence of preparation on math score

sns.catplot(x='test preparation course', y='math score', data=data, kind='boxen', height=8, aspect=1)
#Influence of preparation on writing score

sns.catplot(x='test preparation course', y='writing score', data=data, kind='boxen', height=8, aspect=1)
#Influence of preparation on reading score

sns.catplot(x='test preparation course', y='reading score', data=data, kind='boxen', height=8, aspect=1)
#Distribution of average score for different race/ethnicity groups

sns.catplot(x='race/ethnicity', y='avg', data=data.sort_values(['race/ethnicity']), kind='boxen', height=10, aspect=1.5)
#Distribution of math score for different race/ethnicity groups

sns.catplot(x='race/ethnicity', y='math score', data=data.sort_values(['race/ethnicity']), kind='boxen', height=10, aspect=1.5)
#Distribution of reading score for different race/ethnicity groups

sns.catplot(x='race/ethnicity', y='reading score', data=data.sort_values(['race/ethnicity']), kind='boxen', height=10, aspect=1.5)
#Distribution of writing score for different race/ethnicity groups

sns.catplot(x='race/ethnicity', y='writing score', data=data.sort_values(['race/ethnicity']), kind='boxen', height=10, aspect=1.5)
#Lunch is a very important part of education!!

sns.catplot(x='lunch', y='avg', data=data, kind='boxen', height=8, aspect=1)
#Distributions of average score among different ethnic groups, genders, lunch and preparation types

flatui = ['#4dd2ff','#ff6666']

g = sns.FacetGrid(data, 'lunch', 'test preparation course', height=16, aspect=1)

g.map(sns.violinplot, 'avg', 'race/ethnicity','gender', split=True, palette=sns.color_palette(flatui))

g.add_legend();
#Distribution of average score for different parental levels of education

sns.catplot(x='parental level of education', y='avg', data=data, kind='violin', height=10, aspect=1.5)
#Distribution of math score for different parental levels of education

sns.catplot(x='parental level of education', y='math score', data=data, kind='violin', height=10, aspect=1.5)
#Distribution of reading score for different parental levels of education

sns.catplot(x='parental level of education', y='reading score', data=data, kind='violin', height=10, aspect=1.5)
#Distribution of writing score for different parental levels of education

sns.catplot(x='parental level of education', y='writing score', data=data, kind='violin', height=10, aspect=1.5)