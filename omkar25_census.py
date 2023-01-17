# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib.inline
import seaborn as sns
import pandasql as pds
import warnings
warnings.filterwarnings("ignore")
import matplotlib.mlab as mlab
import scipy.stats as stats

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
Data=pd.read_csv("../input/census2001/all.csv")
Data=Data.fillna(0)
Data=Data.replace('-',0)
Data.describe()
Population=Data['Persons'].sum()
Male_Population=Data['Males'].sum()
Female_Population=Data['Females'].sum()
Female_per_thousand_males=(Female_Population/Male_Population)*1000
print('Female ratio per 1000 males: ',Female_per_thousand_males)
print(Data['Sex.ratio..females.per.1000.males.'].mean())
x=list(range(0,35))
state=list(Data['State'].unique())
state.sort()
xx=Data.groupby('State')[['Persons']+['State']].sum()
plt.bar(x,xx['Persons'])
plt.xticks(x,state, rotation='vertical')
plt.xlabel('States')
plt.ylabel('Population')
plt.show()
plt.figure(figsize=(15,10))
ax=plt.gca()
sns.boxplot(x='State', y='Persons', data=Data, linewidth=1)
ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
plt.title('Population in the "States"')
plt.show()
State_wise_house_condition=Data.groupby('State')['Permanent.House','Semi.permanent.House', 'Temporary.House'].mean()
plt.figure(figsize=(10,8))
sns.heatmap(State_wise_house_condition,cmap='gray_r',annot=True)
plt.show()
age_distribution=Data[['X0...4.years','X5...14.years', 'X15...59.years', 'X60.years.and.above..Incl..A.N.S..']+['State']].groupby('State').sum()
persons=age_distribution.sum(axis=1)
Age_dist=(age_distribution.T/persons.T).T
#plt.figure(figsize=(8,8))
sns.heatmap(Age_dist,annot=True)
plt.title('Age Distribution')
plt.show()
Data_education=Data[['State','District','Primary.school', 'Middle.schools','Secondary.Sr.Secondary.schools', 'College']].fillna(0)
Data_education=Data_education.replace('-',0)
Data_education.head(5)
#Education stats
edu=Data[['Below.Primary', 'Primary', 'Middle','Matric.Higher.Secondary.Diploma', 'Graduate.and.Above']+['State']].groupby('State').sum()
edutotalsum=edu.sum(axis=1)
percentageeducated=(edu.T/edutotalsum.T).T
plt.figure(figsize=(10,8))
sns.heatmap(percentageeducated,cmap='gray_r',linewidth=0.01,annot=True)
plt.title('Heatmap of educated people')
plt.show()
#Literacy Rate
state=Data['State'].unique()
a=list(range(1, len(state)+1))
male_literate_rate=[]
female_literate_rate=[]
for i in state:
    male_literate_rate.append(Data['Males..Literatacy.Rate'][Data['State']==i].mean())
    female_literate_rate.append(Data['Females..Literacy.Rate'][Data['State']==i].mean())
b=[i*2 for i in a]
b1=[x+0.4 for x in b]
b2=[x-0.4 for x in b]
plt.figure(num=None, figsize=(20, 10), dpi=50)
#pt=plt.subplot(111)
plt.bar(b1,male_literate_rate,color='red',align='center')
plt.bar(b2,female_literate_rate,color='blue',align='center')
plt.xticks(b,state, rotation='vertical')
plt.title('Literacy Rate for Each State')
plt.xlabel('State')
plt.ylabel('Literacy Rate', rotation='vertical')
plt.legend(['male_literate_rate','female_literate_rate'])
plt.show()
Data_roadcondition_medical_facility = Data[['Paved.approach.road', 'Mud.approach.road','Primary.Health.Centre', 'Primary.Health.Sub.Centre','State','Bus.services']]
Data_roadcondition_medical_facility[['Paved.approach.road', 'Mud.approach.road','Primary.Health.Centre', 'Primary.Health.Sub.Centre','Bus.services']]=Data_roadcondition_medical_facility[['Paved.approach.road', 'Mud.approach.road','Primary.Health.Centre', 'Primary.Health.Sub.Centre','Bus.services']].apply(pd.to_numeric)
Data_roadcondition_medical_facility_statewise=Data_roadcondition_medical_facility[['Paved.approach.road', 'Mud.approach.road','Primary.Health.Centre', 'Primary.Health.Sub.Centre','Bus.services']+['State']].groupby('State').sum()
#print(Data_roadcondition_medical_facility_statewise.head(5))
#plt.subplot(111)
plt.figure(num=None, figsize=(20, 15), dpi=100)
a=list(range(1, len(state)+1))
b=[x*2 for x in a]
b1=[x-0.33 for x in b]
b2=b
b3=[x+0.33 for x in b]
b4=[x+0.66 for x in b]
b5=[x+0.99 for x in b]
plt.bar(b1,Data_roadcondition_medical_facility_statewise['Paved.approach.road'],color='red',align='center',width=0.24)
plt.bar(b2,Data_roadcondition_medical_facility_statewise['Mud.approach.road'],color='blue',align='center',width=0.25)
plt.bar(b3,Data_roadcondition_medical_facility_statewise['Primary.Health.Centre'],color='green',align='center',width=0.25)
plt.bar(b4,Data_roadcondition_medical_facility_statewise['Primary.Health.Sub.Centre'],color='yellow',align='center',width=0.25)
plt.bar(b5,Data_roadcondition_medical_facility_statewise['Bus.services'],color='purple',align='center',width=0.35)
plt.xticks(b,state, rotation='vertical')
plt.title('Road conditions and Medical Facility availability')
plt.xlabel('State')
plt.ylabel('Total', rotation='vertical')
plt.legend(['Paved Approached Road','Mud Approached Road','Primary Health Centre','Primary Health Sub Centre','Bus Services'])
plt.show()
workers=Data[[ 'Main.workers', 'Marginal.workers', 'Non.workers']+['State']].groupby('State').sum()
total=Data[['Total.workers']+['State']].groupby('State').sum()
#workers_percentage=workers[['Total.workers']].sum(axis=1)
workers_percentage=total.sum(axis=1)
workerspercentage=(workers.T/workers_percentage.T).T
plt.figure(figsize=(10,8))
sns.heatmap(workerspercentage,cmap='gray_r',linewidth=0.01,annot=True)
plt.title('Heatmap of workers')
plt.show()
Area=pd.read_csv('../input/area-state/State_area.csv')
Population=Data[['Persons']+['State']].groupby('State').sum()
Data_area=pds.sqldf('select a.State,a.Persons,b.Area from Population a join Area b on a.State=b.State')
Pop_list=Data_area['Persons'].tolist()
Area=Data_area['Area'].tolist()
Density=[]
for i in range(0,len(Pop_list)):
    Density.append(Pop_list[i]/Area[i])
    plt.subplot(111)
mu=np.mean(Density)
sigma=np.var(Density)
#plt.plot(Density,mlab.normpdf(Density, mu, sigma))
#pdf = stats.norm.pdf(Density, mu, sigma)
#plt.plot(Density,pdf)
plt.hist(Density)
plt.title('Histogram of Population Densities of different states')
plt.xlabel('Population Density')
plt.ylabel('Frequency')
plt.show()
water_Quality=Data[['Drinking.water.facilities', 'Safe.Drinking.water']+['State']].groupby('State').sum()
water_Quality.head(5)
plt.figure(num=None, figsize=(10,8), dpi=100)
a=list(range(1, len(state)+1))
b=[x*2 for x in a]
b1=[x-0.5 for x in b]
b2=b
plt.bar(b1,water_Quality['Drinking.water.facilities'],color='red',align='center',width=0.5)
plt.bar(b2,water_Quality['Safe.Drinking.water'],color='blue',align='center',width=0.5)
plt.xticks(b,state, rotation='vertical')
plt.title('Water Quality')
plt.xlabel('State')
plt.ylabel('Water', rotation='vertical')
plt.legend(['Drinking water facilities','safe Drinking water'])
plt.show()