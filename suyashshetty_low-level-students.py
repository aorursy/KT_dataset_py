# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np #linear algebra

import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pylab as plt

import seaborn as sns

sns.set(style='ticks')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/xAPI-Edu-Data.csv')
# there are no nan values or missing something in 

print(data.info())      
# These are the features names

print(data.columns)     
melt = pd.melt(data,id_vars='Class',value_vars=['raisedhands','VisITedResources','AnnouncementsView'])
sns.swarmplot(x='variable',y='value',hue='Class' , data=melt,palette={'H':'lime','M':'grey','L':'red'})

plt.ylabel('Values from zero to 100')

plt.title('High, middle and low level students')

ave_raisedhands = sum(data['raisedhands'])/len(data['raisedhands'])

ave_VisITedResources = sum(data['VisITedResources'])/len(data['VisITedResources'])

ave_AnnouncementsView = sum(data['AnnouncementsView'])/len(data['AnnouncementsView'])

unsuccess = data.loc[(data['raisedhands'] >= ave_raisedhands) & (data['VisITedResources']>=ave_VisITedResources) & (data['AnnouncementsView']>=ave_AnnouncementsView)  & (data['Class'] == 'L')]

# All features of these two students

print(unsuccess)
data['numeric_class'] = [1 if data.loc[i,'Class'] == 'L' else 2 if data.loc[i,'Class'] == 'M' else 3 for i in range(len(data))]
# Then start with gender: These two are boy so they can be low level due to it :) Girls say YEEESS but lets look

grade_male_ave = sum(data[data.gender == 'M'].numeric_class)/float(len(data[data.gender == 'M']))

grade_female_ave = sum(data[data.gender == 'F'].numeric_class)/float(len(data[data.gender == 'F']))

# Now lets look at nationality

nation = data.NationalITy.unique()

nation_grades_ave = [sum(data[data.NationalITy == i].numeric_class)/float(len(data[data.NationalITy == i])) for i in nation]

data2 = pd.DataFrame({'nation':nation, 'average':nation_grades_ave})

data2.sort_values(by='average', inplace=True, ascending=False)

ax = sns.barplot(x=data2['nation'], y=data2['average'])

for item in ax.get_xticklabels():

    item.set_rotation(90)

jordan_ave = sum(data[data.NationalITy == 'Jordan'].numeric_class)/float(len(data[data.NationalITy == 'Jordan']))

print('Jordan average: '+str(jordan_ave))
# now lets look at topic : chemistry

lessons = data.Topic.unique()

lessons_grade_ave=[sum(data[data.Topic == i].numeric_class)/float(len(data[data.Topic == i])) for i in lessons]

data3 = pd.DataFrame({'lessons':lessons, 'average':lessons_grade_ave}).sort_values(by='average', ascending=True)

ax = sns.barplot(x=data3['lessons'], y=data3['average'])

for item in ax.get_xticklabels():

    item.set_rotation(90)

plt.title('Students Success on different topics')

chemistry_ave = sum(data[data.Topic == 'Chemistry'].numeric_class)/float(len(data[data.Topic == 'Chemistry']))

print('Chemistry average: '+ str(chemistry_ave))
# Lets look at relation with family members

relation = data.Relation.unique()

relation_grade_ave = [sum(data[data.Relation == i].numeric_class)/float(len(data[data.Relation == i])) for i in relation]

ax = sns.barplot(x=relation, y=relation_grade_ave)

plt.title('Relation with father or mother affects success of students')

#Lets look at how many times the student participate on discussion groups

discussion = data.Discussion

discussion_ave = sum(discussion)/len(discussion)

ax = sns.violinplot(y=discussion,split=True,inner='quart')

ax = sns.swarmplot(y=discussion,color='black')

ax = sns.swarmplot(y = unsuccess.Discussion, color='red')

plt.title('Discussion group participation')
# Now lastly lets look at

absence_day = data.StudentAbsenceDays.unique()

absense_day_ave = [sum(data[data.StudentAbsenceDays == i].numeric_class)/float(len(data[data.StudentAbsenceDays == i])) for i in absence_day]

ax = sns.barplot(x=absence_day, y=absense_day_ave)



plt.title('Absence effect on success')
