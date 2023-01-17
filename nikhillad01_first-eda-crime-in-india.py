# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
#importing libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#uploading data
rape_victims=pd.read_csv('../input/crime-analysis/20_Victims_of_rape.csv')
#Rape Victims By State

rape_victims_by_state=rape_victims.groupby('Area_Name').sum()
print('Total Rape Victims: ')
rape_victims_by_state.sort_values(by ='Area_Name',ascending=True).head()
#plotting the Rape Victms by state
plt.subplots(figsize=(15,10))
rt=rape_victims_by_state['Rape_Cases_Reported']
#print(rt)
ax=rt.plot.barh(color='maroon')
ax.set_xlabel("Area Name")
ax.set_ylabel("Number of Rape victims")
ax.set_title("Rape Victims StateWise")
#plt.show()
crime_aginst_womans=pd.read_csv('../input/crime-analysis/42_District_wise_crimes_committed_against_women_2014.csv',error_bad_lines=False)
crime_aginst_womans
crime_type=crime_aginst_womans.groupby("States/UTs").sum()
crime_type.drop("Year",axis=1)
#crime_type.plot.bar()
tota_women=crime_type['Total Crimes against Women']
fig = plt.figure()
az = fig.add_subplot(111)
tota_women.plot.bar(figsize=(15,6),color='maroon')
az.set_title('Crimes against womens')


rw=crime_type['Rape']
kw=crime_type['Kidnapping & Abduction_Total']
dw=crime_type['Dowry Deaths']
sw=crime_type['Sexual Harassment']
mw=crime_type['Murder']


f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True,figsize=(20,8))
#f.subplots_adjust(wspace=0.01, hspace=0)
rw.plot.bar(ax=ax1,color='maroon')
kw.plot.bar(ax=ax2,color='grey')
sw.plot.bar(ax=ax3,color='Burlywood')
mw.plot.bar(ax=ax4,color='pink')
ax1.set_title('Rapes')
ax2.set_title('Kidnappings')
ax3.set_title('Sexual Harassment')
ax4.set_title('Murders')
de=pd.read_csv('../input/crime-analysis/39_Specific_purpose_of_kidnapping_and_abduction.csv')
de
# SPECIFIC PURPOSE FOR KIDNAPPINGS IN MAHARASHTRA
maha_cases=de[de['Area_Name']=="Maharashtra"]
#maha_cases
a=maha_cases.groupby("Sub_Group_Name").sum()
a.drop('Year',axis=1,inplace=True)
a
oo=de
oo
oo['Total_kidnappings'] = oo.sum(axis=1)
oo.fillna(0,inplace=True)
oo.groupby(['Area_Name','Sub_Group_Name'],axis=1).sum()
#ob=oo.groupby('Sub_Group_Name',axis=1).sum()
#ob
#oo.drop('Year',axis=1,inplace=True)
all_india_kidnappings=oo['Total_kidnappings']
oo
ad=a['K_A_Cases_Reported']
plt.subplots(figsize = (15, 6))
av=ad.plot.barh(color='maroon')
av.set_title("Kidnapping and Aduction for specific purposes In Maharashtra")
de
kid_for_specific=de.groupby('Sub_Group_Name').sum()
kid_for_specific.drop('Year',axis=1,inplace=True)
kid_for_specific
kid_for_specific.drop(kid_for_specific.index[13],inplace=True)
kid_for_specific['Total'] = kid_for_specific.sum(axis=1)
kid_for_specific
total_kid=kid_for_specific['Total']
fig = plt.figure()
ab = fig.add_subplot(111)
total_kid.plot.barh(figsize=(15,9),color='Burlywood',ax=ab)
total_kid.set_axis=ab
ab.set_title("Kidnapping for specific purpose ALL over India")
print('Total Kidnappings')
total_kid
all_crimes=pd.read_csv('../input/crime-analysis/01_District_wise_crimes_committed_IPC_2014.csv')
#all_crimes.head()
state_crimes=all_crimes.groupby('States/UTs').sum()
state_crimes.drop('Year', axis = 1, inplace = True)
#state_crimes[['Murder','Rape']].plot.bar()
stt=state_crimes[['Murder','Rape','Rape_Gang Rape','Kidnapping & Abduction_Total','Dacoity','Robbery','Riots']]
#plt.subplots(figsize = (16, 6))
sa=state_crimes[['Murder','Rape','Rape_Gang Rape','Kidnapping & Abduction_Total','Dacoity','Robbery','Riots']]
sa.plot.barh(figsize = (12, 9))
ipc14=pd.read_csv('../input/crime-analysis/01_District_wise_crimes_committed_IPC_2014.csv')
#total Crimes in each state
grouping_state_crimes=ipc14.groupby('States/UTs').sum()
grouping_state_crimes
total_state_crime=grouping_state_crimes['Total Cognizable IPC crimes']
fig = plt.figure()
ax = fig.add_subplot(111)
total_state_crime.plot.bar(figsize=(15,9),color='Burlywood',ax=ax)
total_state_crime.set_axis=ax
ax.set_title("Overall crimes in all States")
# Number of Major crimes state wise
rape14=grouping_state_crimes['Rape']
riots14=grouping_state_crimes['Riots']
murder14=grouping_state_crimes['Murder']
kidnapping14=grouping_state_crimes['Kidnapping & Abduction_Total']

fig = plt.figure()
ax = fig.add_subplot(111)
rape14.plot.bar(color='Red',ax=ax,position=0,width=0.2,figsize = (15, 9))
riots14.plot.bar(color='blue',ax=ax,position=1,width=0.2,figsize = (15, 9))
murder14.plot.bar(color='maroon',ax=ax,position=2,width=0.2,figsize = (15, 9))
kidnapping14.plot.bar(color='darkgrey',ax=ax,position=3,width=0.2,figsize = (15,9))

ax.set_title('Major Crimes in All states')
ax.legend(['Rapes','Riots','Murders','Kidnapping'])
all_india_murders=grouping_state_crimes['Murder'].sum()
all_india_rapes=grouping_state_crimes['Rape'].sum()
all_india_riots=grouping_state_crimes['Riots'].sum()
all_india_kidnapping=grouping_state_crimes['Kidnapping & Abduction_Total'].sum()
all_india_Dowry_Deaths=grouping_state_crimes['Dowry Deaths'].sum()
all_india_acid=grouping_state_crimes['Acid attack'].sum() + grouping_state_crimes['Attempt to Acid Attack'].sum()
all_india_extortion=grouping_state_crimes['Extortion'].sum()
print("Total crimes in India 2014")
print('Murder cases: ',all_india_murders)
print("Rape Cases",all_india_rapes)
print('Riots',all_india_riots)
print('Kidnappings:',all_india_kidnapping)
print('Deaths due to dowry isseue',all_india_Dowry_Deaths)
print('Acid Attacks :',all_india_acid)
print('Extortion Cases :',all_india_extortion)
all_india_murders=grouping_state_crimes['Murder']
all_india_rapes=grouping_state_crimes['Rape']
labels = ['Murder cases','Extortion','Rape Cases','Riots','Deaths due to dowry isseue','Kidnappings']
y = [68268,16420,77356,132412,16916,156824]
#plt.plot.bar(x,y)
fig1, ax1 = plt.subplots(figsize=(15,6))
ax1.pie(y,  labels=labels, autopct='%1.1f%%',
         startangle=90,shadow=True)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title("Overall Crimes in India")
my_circle=plt.Circle( (0,0), 0.7, color='white')#add a circle at the center
p=plt.gcf()
p.gca().add_artist(my_circle)

plt.show()
