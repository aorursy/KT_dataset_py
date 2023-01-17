# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#lifaesave: https://chrisalbon.com/python/data_wrangling/pandas_dataframe_count_values/

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#name data
df = pd.read_csv("/kaggle/input/Granite Curling Club Culture_scrub.csv")

#check head of dataset 
df.head()
df= df.replace(['Strongly Agree'], 5)
df= df.replace(['Agree'], 4)
df= df.replace(['Neutral'], 3)
df= df.replace(['Disagree'], 2)
df= df.replace(['Strongly Disagree'], 1)
df= df.replace(['No Opinion'], 0)

df.head()
df.mean().plot(kind='bar', figsize=(7,5))
#df.mean(axis=0)

#title
plt.title('Average Ratings Accross all Survey Respondents from No Opinion (0) to Strongly Agree (5)')
plt.show()
#create a second df that counts number of response types 

result = df.apply(pd.value_counts).fillna(0); result

df_counts = result.iloc[0:6]

df_counts
#sort participants into groups based on membership length and save them in new dataframes to work with 

gk = df.groupby('Membership length')

# Let's print the first entries 
# in all the groups formed. 

#create a df for novices only 
df_new = gk.get_group('This is my first year') 

#create a df for novices only 
df_novice = gk.get_group('1 - 2 years') 

#create a df for novices only 
df_mid = gk.get_group('3 - 5 years') 

#create a df for novices only 
df_old = gk.get_group('6 - 10 years') 

#create a df for novices only 
df_expert = gk.get_group('More than 11 years') 

#df_new.head()
#look at new members only 

r = df_new.apply(pd.value_counts).fillna(0); result
df_new_counts = r.iloc[0:6]
#look at novice members only 

r = df_novice.apply(pd.value_counts).fillna(0); result
df_novice_counts = r.iloc[0:6] 
# look at mid members only 

r = df_mid.apply(pd.value_counts).fillna(0); result
df_mid_counts = r.iloc[0:6]
# look at more experienced members only 

r = df_old.apply(pd.value_counts).fillna(0); result
df_old_counts = r.iloc[0:6]
# look at more Expert members only

r = df_expert.apply(pd.value_counts).fillna(0); result
df_expert_counts = r.iloc[0:6]

df_expert_counts 
#look at volunteers only

gk = df.groupby('currently volunteering')

df_volunteers = gk.get_group('Yes') 

df_notvolunteers = gk.get_group('No') 
#grouping by volunteer or not, this will be useful later in analysis 
#see volunteers' years in the club

dups_volunteers = df_volunteers.pivot_table(index=['Membership length'], aggfunc='size')
print (dups_volunteers)

#see non-volunteers' years in the club
dups_notvolunteers =df_notvolunteers.pivot_table(index=['Membership length'], aggfunc='size')
print(dups_notvolunteers)
# volunteers bar

volunteers = [13, 17,20,34,43]
index = ['This is my first year','1-2 years', '3-5 years', '6-10 years', '11+ years']

df_vol = pd.DataFrame({'volunteers': volunteers}, index=index)

ax = df_vol.plot.bar(color= '#6BC8ED', figsize=(8,5),rot=0, legend=None)
plt.title('GCC Volunteers by Membership Length')
plt.ylabel('number of responses')
plt.xlabel('n = 127')
#non volunteers bar 

notvolunteers = [11,20,17,15, 27]

index = ['This is my first year','1-2 years', '3-5 years', '6-10 years', '11+ years']

df_novol = pd.DataFrame({'non-volunteers': notvolunteers}, index=index)

ax = df_novol.plot.bar(color= '#6BC8ED', figsize=(8,5),rot=0, legend=None)
plt.title('GCC Non-Volunteers by Membership Length')
plt.ylabel('number of responses')
plt.xlabel('n = 90')  
#assign colors for bar chart
#my_colors = '#081937', '#1468E2','#6BC8ED','#D0E8F5', '#ADEBD1', '#05AA65'

#stacked bar
df_counts[['involving new members','members care','strong friendships','feel welcome']].T.plot.bar(stacked=True, color=my_colors, figsize=(8,5))
plt.title('GCC Approachability')
plt.ylabel('number of responses')
plt.xlabel('n = 220')

#wrap text for x labels 

positions = 0,1,2,3
from textwrap import wrap
labels=['GCC does a good job of involving new members','GCC members care about eachother','I have strong friendships at the GCC','I feel welcome at the GCC']
labels = [ '\n'.join(wrap(l, 15)) for l in labels ]

plt.xticks(positions, labels)

#rotate labels
plt.xticks(rotation = 360)



#rename x labels
labels = ('No Opinion','Strongly Disagree','Disagree','Neutral','Agree','Strongly Agree')
plt.legend
plt.legend((labels), loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.show()
# Is there a difference in sentament for volunteers v non volunteers? Let's breka it down... 

#volunteers stacked bar
df_volunteers_counts[['involving new members','members care','strong friendships','feel welcome']].T.plot.bar(stacked=True, color=my_colors, figsize=(8,5))

#labels
plt.title('GCC Approachability Ratings for Volunteers')
plt.ylabel('number of responses')
plt.xlabel('n = 127')

#wrap text for x labels 
positions = 0,1,2,3
from textwrap import wrap
labels=['GCC does a good job of involving new members','GCC members care about eachother','I have strong friendships at the GCC','I feel welcome at the GCC']
labels = [ '\n'.join(wrap(l, 15)) for l in labels ]
plt.xticks(positions, labels)

#rotate labels
plt.xticks(rotation = 360)

#rename legend
labels = ('No Opinion','Strongly Disagree','Disagree','Neutral','Agree','Strongly Agree')
plt.legend
plt.legend((labels), loc='center left', bbox_to_anchor=(1.0, 0.5))


#non volunteers stacked bar
df_notvolunteers_counts[['involving new members','members care','strong friendships','feel welcome']].T.plot.bar(stacked=True, color=my_colors, figsize=(8,5))

#labels
plt.title('GCC Approachability Ratings for Non-Volunteers')
plt.ylabel('number of responses')
plt.xlabel('n = 90')

#wrap text for x labels 
positions = 0,1,2,3
from textwrap import wrap
labels=['GCC does a good job of involving new members','GCC members care about eachother','I have strong friendships at the GCC','I feel welcome at the GCC']
labels = [ '\n'.join(wrap(l, 15)) for l in labels ]
plt.xticks(positions, labels)

#rotate labels
plt.xticks(rotation = 360)

#rename legend
labels = ('No Opinion','Strongly Disagree','Disagree','Neutral','Agree','Strongly Agree')
plt.legend
plt.legend((labels), loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.show()
#Might be fun to look at the means of each answer, according to membership length 
r1 = "first year", df_new[['involving new members','members care','strong friendships','feel welcome']].mean()
r2 = 'novice', df_novice[['involving new members','members care','strong friendships','feel welcome']].mean()
r3 = 'mid', df_mid[['involving new members','members care','strong friendships','feel welcome']].mean()
r4 = 'old', df_old[['involving new members','members care','strong friendships','feel welcome']].mean()
r5 = 'expert', df_expert[['involving new members','members care','strong friendships','feel welcome']].mean()

print(r1, r2, r3, r4, r5)
#stacked bar graphs related to culture by years of club membership

#new
df_new_counts [['involving new members','members care','strong friendships','feel welcome']].T.plot.bar(stacked=True,color=my_colors, figsize=(8,5))
plt.title('GCC Approachability - First years')
plt.ylabel('number of responses')
plt.xlabel('n = 24')

positions = 0,1,2,3

#wrap text for x labels 
from textwrap import wrap
labels=['GCC does a good job of involving new members','GCC members care about eachother','I have strong friendships at the GCC','I feel welcome at the GCC']
labels = [ '\n'.join(wrap(l, 15)) for l in labels ]

plt.xticks(positions, labels)

#rotate labels
plt.xticks(rotation = 360)


#rename x labels
labels = ('No Opinion','Strongly Disagree','Disagree','Neutral','Agree','Strongly Agree')
plt.legend
plt.legend((labels), loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.show()

#novice
df_novice_counts [['involving new members','members care','strong friendships','feel welcome']].T.plot.bar(stacked=True,color=my_colors, figsize=(8,5))
plt.title('GCC approachability ratings for early members (1-2 years)')
plt.ylabel('number of responses')
plt.xlabel('n = 37')

positions = 0,1,2,3

#wrap text for x labels 
from textwrap import wrap
labels=['GCC does a good job of involving new members','GCC members care about eachother','I have strong friendships at the GCC','I feel welcome at the GCC']
labels = [ '\n'.join(wrap(l, 15)) for l in labels ]

plt.xticks(positions, labels)

#rotate labels
plt.xticks(rotation = 360)


#rename x labels
labels = ('No Opinion','Strongly Disagree','Disagree','Neutral','Agree','Strongly Agree')
plt.legend
plt.legend((labels), loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.show()


#mid_level
df_mid_counts [['involving new members','members care','strong friendships','feel welcome']].T.plot.bar(stacked=True,color=my_colors, figsize=(8,5))

plt.title('GCC approachability ratings for mid-experience members (3-5 years)')
plt.ylabel('number of responses')
plt.xlabel('n = 37')

#wrap text for x labels 

positions = 0,1,2,3

from textwrap import wrap
labels=['GCC does a good job of involving new members','GCC members care about eachother','I have strong friendships at the GCC','I feel welcome at the GCC']
labels = [ '\n'.join(wrap(l, 15)) for l in labels ]

plt.xticks(positions, labels)

#rotate labels
plt.xticks(rotation = 360)

#rename  and position legend 
labels = ('No Opinion','Strongly Disagree','Disagree','Neutral','Agree','Strongly Agree')
plt.legend
plt.legend((labels), loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.show()


#mid
df_old_counts[['involving new members','members care','strong friendships','feel welcome']].T.plot.bar(stacked=True,color=my_colors, figsize=(8,5))

plt.title('GCC approachability ratings for experienced members (6-10 years)')
plt.ylabel('number of responses')
plt.xlabel('n = 51')

#wrap text for x labels 
from textwrap import wrap
labels=['GCC does a good job of involving new members','GCC members care about eachother','I have strong friendships at the GCC','I feel welcome at the GCC']
labels = [ '\n'.join(wrap(l, 15)) for l in labels ]

positions = 0,1,2,3
plt.xticks(positions, labels)

#rotate labels
plt.xticks(rotation = 360)

#rename  and position legend 
labels = ('No Opinion','Strongly Disagree','Disagree','Neutral','Agree','Strongly Agree')
plt.legend
plt.legend((labels), loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.show()

#old
df_expert_counts[['involving new members','members care','strong friendships','feel welcome']].T.plot.bar(stacked=True,color=my_colors, figsize=(8,5))

plt.title('GCC approachability ratings for long-time members (11+ years)')
plt.ylabel('number of responses')
plt.xlabel('n = 71')

#x axis labels 
positions = (0,1,2,3)

#wrap text for x labels 
from textwrap import wrap
labels=['GCC does a good job of involving new members','GCC members care about eachother','I have strong friendships at the GCC','I feel welcome at the GCC']
labels = [ '\n'.join(wrap(l, 15)) for l in labels ]

plt.xticks(positions, labels)

#rotate labels
plt.xticks(rotation = 360)

#rename and position legend 
labels = ('No Opinion','Strongly Disagree','Disagree','Neutral','Agree','Strongly Agree')
plt.legend
plt.legend((labels), loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.show()
# frequency count of results - questions related to awareness and access to volunteer roles 

#stacked bar
df_counts[['involve all interests, skills and availability', 'communicating to members', 'understand why GCC needs volunteers']].T.plot.bar(stacked=True, color=my_colors, figsize=(8,5))
plt.title('Awareness and Access to Volunteer Roles')
plt.ylabel('number of responses')
plt.xlabel('n = 220')

#x axis labels 
positions = (0,1,2)

#wrap text for x labels 
from textwrap import wrap
labels=['GCC involves all members in projects according to interests, skills availability', 'GCC does a good job of communicating to members', 'I understand why GCC needs volunteers']
labels = [ '\n'.join(wrap(l, 15)) for l in labels ]

plt.xticks(positions, labels)

#rotate labels
plt.xticks(rotation = 360)

#rename legend labels
labels = ('No Opinion','Strongly Disagree','Disagree','Neutral','Agree','Strongly Agree')
plt.legend
plt.legend((labels), loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.show()
# frequency count of results - questions related to awareness and access to volunteer roles broken down by membership length

# these all look pretty similar so I will not break this down for the final report

#new graph
#df_new_counts[['involve all interests, skills and availability', 'communicating to members','understand why GCC needs volunteers']].T.plot.bar(stacked=True)

#novice graph
#df_novice_counts[['involve all interests, skills and availability', 'communicating to members','understand why GCC needs volunteers']].T.plot.bar(stacked=True)

#mid 
#df_mid_counts[['involve all interests, skills and availability', 'communicating to members','understand why GCC needs volunteers']].T.plot.bar(stacked=True)

#old graph

#df_old_counts[['involve all interests, skills and availability', 'communicating to members','understand why GCC needs volunteers']].T.plot.bar (stacked=True,color=my_colors, figsize=(8,5))
#plt.title('GCC approachability ratings for long-time members (11+ years)')
#plt.ylabel('number of responses')
#plt.xlabel('n = 37')

#expert graph
#df_expert[['involve all interests, skills and availability', 'communicating to members','understand why GCC needs volunteers']].mean()

#df_expert_counts[['involve all interests, skills and availability', 'communicating to members','understand why GCC needs volunteers']].T.plot.bar (stacked=True,color=my_colors, figsize=(8,5))

#plt.title('GCC approachability ratings for long-time members (11+ years)')
#plt.ylabel('number of responses')
#plt.xlabel('n = 37')

#rename  and position legend 
# labels = ('No Opinion','Strongly Disagree','Disagree','Neutral','Agree','Strongly Agree')
#plt.legend
#plt.legend((labels), loc='center left', bbox_to_anchor=(1.0, 0.5))

#plt.show()
#mulit select is a doozy - google forms does a good job of this - see google forms for responses
# ['prevent you from volunteering']
# ['lack of info prevented volunteering']
# frequency count of results - questions related to documentation

#stacked bar
df_counts[['update processes and rules']].T.plot.bar(stacked=True, color=my_colors, figsize=(8,5))
plt.title('GCC works to update processes and rules to meet the needs of its members')
plt.ylabel('number of responses')
plt.xlabel('n = 220')
plt.xticks([])


#rename x labels
labels = ('No Opinion','Strongly Disagree','Disagree','Neutral','Agree','Strongly Agree')
plt.legend
plt.legend((labels), loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.show()
# frequency counts of questions related to training and growth potential 

#stacked bar
df_counts[['use my skills']].T.plot.bar(stacked=True, color=my_colors, figsize=(8,5))
plt.title('GCC provides me with opportunities to use my skills and talents')
plt.ylabel('number of responses')
plt.xlabel('n = 220')
plt.xticks([])


#rename x labels
labels = ('No Opinion','Strongly Disagree','Disagree','Neutral','Agree','Strongly Agree')
plt.legend
plt.legend((labels), loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.show()
# Mulit select 
#'skills are inadequate to volunteer','types of skills you need to volunteer' - see google forms
# frequency counts of questions related to expectations of roles & repsponsabilities as well as burnout 


#stacked bar
df_counts[['Volunteer opportunities- reasonable amount of my time', 'Volunteer opportunities- reasonable number of projects',
           'Volunteer opportunities - well organized]']].T.plot.bar(stacked=True, color=my_colors, figsize=(8,5))


plt.title('Volunteering & Perception of Roles & Reponsability')
plt.ylabel('number of responses')

#x axis labels 
positions = (0,1,2)

#wrap text for x labels 
from textwrap import wrap
labels=['reasonable amount of my time','reasonable number of projects', 'well organized']
labels = [ '\n'.join(wrap(l, 10)) for l in labels ]

plt.xticks(positions, labels)

#rotate labels
plt.xticks(rotation = 360)

#rename legend labels
labels = ('No Opinion','Strongly Disagree','Disagree','Neutral','Agree','Strongly Agree')
plt.legend
plt.legend((labels), loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.show()

#frequency counts related to volunteer motivations

#stacked bar
df_counts[['Volunteering makes a positive impact']].T.plot.bar(stacked=True, color=my_colors, figsize=(8,5))
plt.title('Volunteering makes a positive impact')
plt.ylabel('number of responses')
plt.xlabel('n = 220')
plt.xticks([])


#rename x labels
labels = ('No Opinion','Strongly Disagree','Disagree','Neutral','Agree','Strongly Agree')
plt.legend
plt.legend((labels), loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.show()
