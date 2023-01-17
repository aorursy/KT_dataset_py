#Import libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
%matplotlib inline
#Read data from "multipleChoiceResponses.csv" file
data=pd.read_csv('../input/multipleChoiceResponses.csv', skiprows=[1])
df = pd.DataFrame(data)
#Select column 'Q6' and select those who identify as "data scientists"
occupation = df['Q6'].value_counts()
total = occupation.sum()
data_positive = occupation['Data Scientist']
data_negative = total - data_positive

plt.figure(1, figsize=(14,10))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 0], aspect=1, title='Are you a Data Scientist?')
plt.axis('equal')
labels = 'Yes (Positive)', 'No (Negative)'
sizes = [data_positive, data_negative]
colors = ['#d9b37c','#B94E8A']
explode = (0.15, 0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)

plt.subplot(the_grid[0, 1], aspect=1, title='If not, what is your occupation?')
num_da = occupation['Data Analyst']
num_se = occupation['Software Engineer']
num_st = occupation['Student']
num_ne = occupation['Not employed']
num_other = (data_negative - (num_da + num_se + num_st + num_ne))
plt.axis('equal')
labels = 'Data Analyst', 'Software Engineer', 'Student', 'Other', 'Not Employed(*)'
sizes = [num_da, num_se, num_st, num_other, num_ne]
colors = ['#FFD7F1', '#D87CA1', '#B94E8A', '#7D156D', '#fff5ff']
explode = (0, 0, 0, 0, 0.4)

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=360)

plt.suptitle('2018 Kaggle ML & DS Survey Challenge Results', fontsize=22)

plt.show()
# from now on, n = 22,900
n = df[df['Q6'].notnull()]
# Question: "Are you a Data Scientist"?
# Yes - positive population
pos_pop = n.loc[n['Q6'] == 'Data Scientist']
# No - negative population
neg_pop = n.loc[n['Q6'] != 'Data Scientist'] 
agegroup = n['Q2']
age = agegroup.value_counts().sort_index()

sns.set(style="white", context="talk")
sns.color_palette("Set2")

# Set up the matplotlib figure
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=False)
x1 = age.index
y1 = age
sns.barplot(x=x1, y=y1, ax=ax1)
ax1.set_title("Age Group of Participants")
ax1.axhline(0, color="k", clip_on=False)
ax1.set_ylabel("Number of ppl")
plt.subplots_adjust(hspace = 0.3)

#question - 'are you a data scientist'? - age group
# "no" negative
y2 = neg_pop['Q2'].value_counts().sort_index() 
# "yes" positive
y3 = pos_pop['Q2'].value_counts().sort_index()

p1 = plt.bar(x1,y2,color="#B94E8A")
p2 = plt.bar(x1,y3,color="#d9b37c")
plt.plot(2,1400,'wd', markersize=16)
plt.text(2, 650, '33.09%', horizontalalignment='center', color="w", size=12)
plt.text(1, 300, '17.26%', horizontalalignment='center', color="w", size=12)
plt.text(3, 400, '21.1%', horizontalalignment='center', color="w", size=12)

plt.ylabel('Number of ppl')
plt.legend((p2[0], p1[0]), ('Yes', 'No'))
ax2.axhline(0, color="k", clip_on=False)
plt.title("Are you a Data Scientist?")
plt.show()
yes_percentage = []
for x in range(0,12): 
    yes_percentage.append(str(round((y3[x]/4137*100),2)) + "%")
    
#question - 'are you a data scientist'? - gender
# "no" negative
y4 = neg_pop['Q1'].value_counts().sort_index() 
# "yes" positive
y5 = pos_pop['Q1'].value_counts().sort_index()

gender_group = n['Q1']
gender = gender_group.value_counts().sort_index()
x2 = gender.index

plt.figure(1, figsize=(14,10))
the_grid = GridSpec(3, 3)

plt.subplot(the_grid[0, 0], aspect=1, title='Gender of Participants')
plt.axis('equal')
labels = 'Female', 'Male', '', ''
sizes = gender
colors = ['#FFBE00', '#005874', '#E6E6D4', '#E6E6D4']
explode = (0.15,0,0,0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors, shadow=True, startangle=140)
plt.text(-1,0.08, '16.8%', size=12)
# 16.8% were women

plt.subplot(the_grid[0, 1], aspect=1, title='Gender of Data Scientists')
plt.axis('equal')
sizes = y5
colors = ['#FFBE00', '#005874', '#E6E6D4', '#E6E6D4']
plt.pie(sizes, labels=labels, colors=colors, shadow=True, explode=explode, startangle=130)
plt.text(-0.98,0.2, '16.6%', size=12)

plt.subplot(the_grid[0, 2], aspect=1, title='Gender of Non-Data Scientists')
plt.axis('equal')
sizes = y4
colors = ['#FFBE00', '#005874', '#E6E6D4', '#E6E6D4']
plt.pie(sizes, labels=labels, colors=colors, shadow=True, explode=explode, startangle=130)
plt.text(-0.98, 0.2, '16.9%', size=12)
plt.show()
# correlations between undergrad studies and occupation
data_uni = n.groupby(['Q5','Q6'])['Q6'].count().to_frame(name = 'count').reset_index()
# some participants did not answer the question in the survey
#data_uni.fillna('Unknown', inplace=True)
data_scientist = data_uni.loc[data_uni['Q6'] == 'Data Scientist']
data_scientist = data_scientist.sort_values('count', ascending=False)
business = data_uni.loc[data_uni['Q5'] == 'A business discipline (accounting, economics, finance, etc.)']
business = business.sort_values('count', ascending=False)

#plt.figure(1, figsize=(14,10))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 12))
sns.barplot(x="count", y="Q5", data=data_scientist, ax=ax2)
sns.barplot(x="count", y="Q6", data=business, ax=ax1)

ax1.set_ylabel('')    
ax1.set_xlabel('(total=1,760)')
ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax1.invert_xaxis()
ax1.set_title("Business Majors")
ax1.text(122, 0,'Data Scientist (291)', verticalalignment='center',fontsize=14, bbox=dict(facecolor='red'))
ax1.text(118, 1,'Data Analyst (267)', verticalalignment='center',fontsize=14)
ax1.text(142, 3,'Business Analyst (205)', verticalalignment='center',fontsize=14)
ax1.text(120, 14,'Data Engineer (28)', verticalalignment='center',fontsize=14)
ax1.text(168, 18,'DBA/Database Engineer (5)', verticalalignment='center',fontsize=14)
ax1.text(114, 20,'Data Journalist (2)', verticalalignment='center',fontsize=14)

ax2.set_ylabel('')    
ax2.set_xlabel('(total=4,115)')
ax2.set_yticklabels([])
ax2.set_xticklabels([])
ax2.set_title("Data Scientists")

ax2.text(5, 0,'Computer Science (1403)', verticalalignment='center',fontsize=14)
ax2.text(5, 1,'Mathmatics/Statistics (835)', verticalalignment='center',fontsize=14)
ax2.text(5, 2,'Engineering (645)', verticalalignment='center',fontsize=14)
ax2.text(5, 3,'Physics/Astronomy (339)', verticalalignment='center',fontsize=14)
ax2.text(5, 4,'Business (291)', verticalalignment='center',fontsize=14, bbox=dict(facecolor='red'))
# Data Scientists recommend this language for aspiring newcomers
rec_lan = pos_pop['Q18'].value_counts()
x3 = rec_lan.index
str(round((rec_lan[0]/data_positive*100),2))+"% of the data scientists said that they recommend Python for aspiring data scientists to learn first."
# how many data scientists are 'self-taught'?
self_taught = pos_pop['Q35_Part_1'].dropna()
self_taught = self_taught.sort_values(ascending=False)
self_taught = self_taught[self_taught >= 50.0]
online_course = pos_pop['Q35_Part_2'].dropna()
online_course = online_course.sort_values(ascending=False)
online_course = online_course[online_course >= 50.0] 

st = self_taught.describe().to_frame().rename(columns={"Q35_Part_1": "Self-taught"})
oc = online_course.describe().to_frame().rename(columns={"Q35_Part_2": "Online Course"})

frames = [st,oc]
pd.concat(frames, axis=1)
# Asking the Data Scientists - During a typical data science 
# project at work or school, approximately what proportion of your time is devoted to the following? 

plt.figure(1, figsize=(15,10))
plt.subplot(2,3,1) 
ax1 = sns.boxplot(y=pos_pop['Q34_Part_1'], palette="Set3")
ax1.set_ylabel("proportion of time devoted(%)")
ax1.set_xlabel("Gathering data")
plt.subplot(2,3,2)
ax2 = sns.boxplot(y=pos_pop['Q34_Part_2'], palette="Set3")
ax2.set_ylabel("")
ax2.set_xlabel("Cleaning data")
plt.subplot(2,3,3)
ax3 = sns.boxplot(y=pos_pop['Q34_Part_3'], palette="Set3")
ax3.set_ylabel("")
ax3.set_xlabel("Visualizing data")
plt.subplot(2,3,4)
ax4 = sns.boxplot(y=pos_pop['Q34_Part_4'], palette="Set3")
ax4.set_ylabel("proportion of time devoted(%)")
ax4.set_xlabel("Model building/selection")
plt.subplot(2,3,5)
ax5 = sns.boxplot(y=pos_pop['Q34_Part_5'], palette="Set3")
ax5.set_ylabel("")
ax5.set_xlabel("Model production")
plt.subplot(2,3,6)
ax6 = sns.boxplot(y=pos_pop['Q34_Part_6'], palette="Set3")
ax6.set_ylabel("")
ax6.set_xlabel("Finding insights & communicating")