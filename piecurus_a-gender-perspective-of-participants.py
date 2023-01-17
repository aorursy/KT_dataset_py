# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set()
# Any results you write to the current directory are saved as output.

d = pd.read_csv("../input/Speed Dating Data.csv", encoding="ISO-8859-1")
g = sns.factorplot(y="wave", hue='gender',data=d, size=10, kind="count", palette="muted", orient='h')
g = sns.factorplot(y="wave", x="age", hue='gender',data=d, size=10, kind="bar", palette="muted", orient='h')
field_cd = ['Law','Math','Social/Psy','MedicalScience,Pharmaceuticals,BioTech ',
                   'Eng','Eglish/Creative Writing/ Journalism','History/Religion/Philosophy',
                  'Business/Econ/Finance','Education, Academia','Biological Sciences/Chemistry/Physics',
                  'Social Work','Undergrad/undecided','Political Science/International Affairs',
                  'Film','Fine Arts/Arts Administration','Languages','Architecture','Others']
g = plt.figure(figsize=(20,20))
g = sns.countplot(y="field_cd",data=d, hue='gender')
g.set(yticklabels=field_cd)
g = plt.yticks(rotation=0,fontsize=15)
race = ['Black/African American','European/Caucasian-American','Latino/Hispanic American',
        'Asian/Pacific Islander/Asian-American','Native American','Other']
g = plt.figure(figsize=(10,10))
g = sns.countplot(y="race",data=d, hue='gender')
g.set(yticklabels=race)
g = plt.yticks(rotation=0,fontsize=15)
goal = ['Seemed like a fun night out','To meet new people','To get a date','Looking for a serious relationship',
        'To say I did it','Other']
g = plt.figure(figsize=(10,10))
g = sns.countplot(y="goal",data=d, hue='gender')
g.set(yticklabels=goal)
g = plt.yticks(rotation=0,fontsize=15)
go_on_dates =['Several times a week','Twice a week','Once a week','Twice a month',
              'Once a month','Several times a year','Almost never']
g = plt.figure(figsize=(10,10))
g = sns.countplot(y="date",data=d, hue='gender')
g.set(yticklabels=go_on_dates)
g = plt.yticks(rotation=0,fontsize=15)
go_out = ['Several times a week','Twice a week','Once a week','Twice a month',
              'Once a month','Several times a year','Almost never']
g = plt.figure(figsize=(10,10))
g = sns.countplot(y="go_out",data=d, hue='gender')
g.set(yticklabels=go_on_dates)
g = plt.yticks(rotation=0,fontsize=15)
career = ['Lawyer','Academic/Research','Psychologist','Doctor/Medicine','Engineer','Creative Arts/Entertainment',
          'Banking/Consult/Finance/Market/CEO/Entrepr/Admin','Real Estate',
          'International/Humanitarian Affairs','Undecided','Social Work','Speech Pathology','Politics',
          'Pro sports/Athletics','Other','Journalism','Architecture']
g = plt.figure(figsize=(10,10))
g = sns.countplot(y="career_c",data=d, hue='gender')
g.set(yticklabels=career)
g = plt.yticks(rotation=0,fontsize=15)
# for some reason, seaborn doesn't allow to plot the "hue" input when x/y are ont specified
# I'll use a general matplotlib approach

activities_interested=['sports','tvsports','exercise','dining','museums','art','hiking','gaming','clubbing','reading',
                       'tv','theater','movies','concerts','music','shopping','yoga']
temp = d.groupby(['gender']).mean()[activities_interested].values

g = plt.figure(figsize=(15,15))
g = plt.barh(np.arange(0,2*temp.shape[1],2)-0.2,temp[0,:], height=0.5,color=[0,0,1],alpha=0.5,label='Female')
g = plt.barh(np.arange(0,2*temp.shape[1],2)+0.2,temp[1,:], height=0.5,color=[0,1,0],alpha=0.5,label='Male')
g = plt.yticks(np.arange(0,2*temp.shape[1],2)+0.2,activities_interested,fontsize=16)
g = plt.ylim(-1,2*temp.shape[1]+1)
g = plt.legend(loc=0,fontsize=16)
male_data = d[d['gender']==1].groupby('iid').sum()
male_data = male_data[activities_interested+['match']]
gotMatch = np.cast['int16'](male_data['match'])
g = plt.figure(figsize=(10,10))
g = plt.hist(gotMatch,range(20))
g = plt.xlabel('Number of Matches for Guys',fontsize=16)
male_threshold = 8
from sklearn.ensemble import ExtraTreesClassifier

male_data = d[d['gender']==1].groupby('iid').sum()
male_data = male_data[activities_interested+['match']]
gotMatch = np.cast['int16'](male_data['match']>male_threshold)
male_data = male_data.fillna(-1).values[:,:-1]

etc = ExtraTreesClassifier(n_estimators=3333,random_state=0)

etc.fit(male_data,gotMatch)
importances = etc.feature_importances_
indices = np.argsort(importances)[::-1]
g = plt.figure(figsize=(20,20))
g = plt.title("Good Interests for guys", fontsize=16)
g = plt.barh(range(len(activities_interested)), importances[indices],color="b",align="center",alpha=0.6)
g = temp = np.array(activities_interested)
g = plt.yticks(range(len(activities_interested)), temp[indices], fontsize=16)
g = plt.ylim([-1, len(activities_interested)+.1])
female_data = d[d['gender']==0].groupby('iid').sum()
female_data = female_data[activities_interested+['match']]
gotMatch = np.cast['int16'](female_data['match'])
g = plt.figure(figsize=(10,10))
g = plt.hist(gotMatch,range(20))
g = plt.xlabel('Number of Matches for Girls',fontsize=16)
female_threshold = 8
female_data = d[d['gender']==0].groupby('iid').sum()
female_data = female_data[activities_interested+['match']]
gotMatch = np.cast['int16'](female_data['match']>female_threshold)
female_data = female_data.fillna(-1).values[:,:-1]

etc = ExtraTreesClassifier(n_estimators=3333,random_state=0)

etc.fit(female_data,gotMatch)
importances = etc.feature_importances_
indices = np.argsort(importances)[::-1]
g = plt.figure(figsize=(20,20))
g = plt.title("Good Interests for girls", fontsize=16)
g = plt.barh(range(len(activities_interested)), importances[indices],color="g",align="center",alpha=0.6)
g = temp = np.array(activities_interested)
g = plt.yticks(range(len(activities_interested)), temp[indices], fontsize=16)
g = plt.ylim([-1, len(activities_interested)+0.1])
temp = d[d['wave']==18]
w18_male_data = temp[temp['gender']==1].groupby('iid').sum()
w18_male_data = w18_male_data[activities_interested+['match']]
gotMatch = np.cast['int16'](w18_male_data['match'])
g = plt.figure(figsize=(10,10))
g = plt.hist(gotMatch,range(10))
g = plt.xlabel('Number of Matches for Guys in Wave 18',fontsize=16)
male_threshold = 0

gotMatch = np.cast['int16'](w18_male_data['match']>male_threshold)
w18_male_data = w18_male_data.fillna(-1).values[:,:-1]

etc = ExtraTreesClassifier(n_estimators=3333,random_state=0)

etc.fit(w18_male_data,gotMatch)
importances = etc.feature_importances_
indices = np.argsort(importances)[::-1]
g = plt.figure(figsize=(20,20))
g = plt.title("Good Interests for guys in Wave 18", fontsize=16)
g = plt.barh(range(len(activities_interested)), importances[indices],color="b",align="center",alpha=0.6)
g = temp = np.array(activities_interested)
g = plt.yticks(range(len(activities_interested)), temp[indices], fontsize=16)
g = plt.ylim([-1, len(activities_interested)+.1])
temp = d[d['wave']==18]
w18_female_data = temp[temp['gender']==0].groupby('iid').sum()
w18_female_data = w18_female_data[activities_interested+['match']]
gotMatch = np.cast['int16'](w18_female_data['match'])
g = plt.figure(figsize=(10,10))
g = plt.hist(gotMatch,range(10))
g = plt.xlabel('Number of Matches for Girls in Wave 18',fontsize=16)
female_threshold = 0

gotMatch = np.cast['int16'](w18_female_data['match']>female_threshold)
w18_female_data = w18_female_data.fillna(-1).values[:,:-1]

etc = ExtraTreesClassifier(n_estimators=3333,random_state=0)

etc.fit(w18_female_data,gotMatch)
importances = etc.feature_importances_
indices = np.argsort(importances)[::-1]
g = plt.figure(figsize=(20,20))
g = plt.title("Good Interests for girls in Wave 18", fontsize=16)
g = plt.barh(range(len(activities_interested)), importances[indices],color="b",align="center",alpha=0.6)
g = temp = np.array(activities_interested)
g = plt.yticks(range(len(activities_interested)), temp[indices], fontsize=16)
g = plt.ylim([-1, len(activities_interested)+.1])
