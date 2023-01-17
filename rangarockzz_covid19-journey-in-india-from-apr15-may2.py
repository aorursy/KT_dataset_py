# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
hospital_info=pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')[:36]
hospital_info
obj=list(hospital_info.columns[2:8])

for ob in obj:
    hospital_info[ob]=hospital_info[ob].astype(int,errors='ignore')
plt.suptitle('HEALTH INFORMATION STATEWISE',fontsize=20,color='green')
fig = plt.figure(figsize=(30,15)) 
#plt.title()
#four subplots
plt1 = fig.add_subplot(221) 
plt2 = fig.add_subplot(222) 
plt3 = fig.add_subplot(223) 
plt4 = fig.add_subplot(224) 

primary=hospital_info.nlargest(10,'NumPrimaryHealthCenters_HMIS')

plt1.set_title('Primary Health Care Centers')
plt1.barh(primary['State/UT'],primary['NumPrimaryHealthCenters_HMIS'],color ='blue');

community=hospital_info.nlargest(10,'NumCommunityHealthCenters_HMIS')
plt2.set_title('Community Health Centers Info')
plt2.barh(community['State/UT'],community['NumCommunityHealthCenters_HMIS'],color='green')

dist=hospital_info.nlargest(10,'NumDistrictHospitals_HMIS')
plt3.set_title("District Hospitals Info" )
plt3.barh(dist['State/UT'],dist['NumDistrictHospitals_HMIS'],color='gold')

subd=hospital_info.nlargest(10,'TotalPublicHealthFacilities_HMIS')
plt4.set_title('PUblic Health Facilities Info')
plt4.barh(subd['State/UT'],subd['TotalPublicHealthFacilities_HMIS'],color='violet')

fig.subplots_adjust(hspace=.5,wspace=0.2) 
apr_15=pd.read_csv('../input/dynamiccovid19india-statewise/15-04-2020.csv')[:33]
apr_16=pd.read_csv('../input/dynamiccovid19india-statewise/16-04-2020.csv')[:33]
apr_17=pd.read_csv('../input/dynamiccovid19india-statewise/17-04-2020.csv')[:33]
#mar_18=pd.read_csv('18-04-2020.csv')[:33]
apr_19=pd.read_csv('../input/dynamiccovid19india-statewise/19-04-2020.csv')[:33]
apr_20=pd.read_csv('../input/dynamiccovid19india-statewise/20-04-2020.csv')[:33]
apr_21=pd.read_csv('../input/dynamiccovid19india-statewise/21-04-2020.csv')[:33]

apr_22=pd.read_csv('../input/dynamiccovid19india-statewise/22-04-2020.csv')[:32]
apr_23=pd.read_csv('../input/dynamiccovid19india-statewise/23-04-2020.csv')[:32]

apr_24=pd.read_csv('../input/dynamiccovid19india-statewise/24-04-2020.csv')[:32]
apr_25=pd.read_csv('../input/dynamiccovid19india-statewise/25-04-2020.csv')[:32]

apr_26=pd.read_csv('../input/dynamiccovid19india-statewise/26-04-2020.csv')[:32]
apr_27=pd.read_csv('../input/dynamiccovid19india-statewise/27-04-2020.csv')[:32]
apr_28=pd.read_csv('../input/dynamiccovid19india-statewise/28-04-2020.csv')[:32]
apr_29=pd.read_csv('../input/dynamiccovid19india-statewise/29-04-2020.csv')[:32]
apr_30=pd.read_csv('../input/dynamiccovid19india-statewise/30-04-2020.csv')[:32]
may_1=pd.read_csv('../input/dynamiccovid19india-statewise/01-05-2020.csv')[:32]
may_2=pd.read_csv('../input/dynamiccovid19india-statewise/02-05-2020.csv')[:32]
dates={'dates':['15/4/2020','16/4/2020','17/4/2020','18/4/2020','20/4/2020','21/4/2020','22/4/2020','23/4/2020','24/4/2020',
                '25/4/2020','26/4/2020','27/4/2020','28/4/2020','29/4/2020','30/4/2020','01/05/2020','02/05/2020'],
       'cases':[
apr_15['Total Confirmed cases (Including 76 foreign Nationals)'].sum(),
apr_16['Total Confirmed cases (Including 76 foreign Nationals)'].sum(),
apr_17['Total Confirmed cases (Including 76 foreign Nationals)'].sum(),
apr_19['Total Confirmed cases (Including 77 foreign Nationals)'].sum(),
apr_20['Total Confirmed cases (Including 77 foreign Nationals)'].sum(),
apr_21['Total Confirmed cases (Including 77 foreign Nationals)'].sum(),
apr_22['Total Confirmed cases (Including 77 foreign Nationals)'].sum(),
apr_23['Total Confirmed cases (Including 77 foreign Nationals)'].sum(),
apr_24['Total Confirmed cases (Including 77 foreign Nationals)'].sum(),
apr_25['Total Confirmed cases (Including 111 foreign Nationals)'].sum(),
apr_26['Total Confirmed cases (Including 111 foreign Nationals)'].sum(),
apr_27['Total Confirmed cases (Including 111 foreign Nationals)'].sum(),
apr_28['Total Confirmed cases (Including 111 foreign Nationals)'].sum(),
apr_29['Total Confirmed cases (Including 111 foreign Nationals)'].sum(),
apr_30['Total Confirmed cases (Including 111 foreign Nationals)'].sum(),
may_1['Total Confirmed cases (Including 111 foreign Nationals)'].sum(),
may_2['Total Confirmed cases (Including 111 foreign Nationals)'].sum()],
      }
dates=pd.DataFrame(data=dates,index=range(17))
dates.head()
import numpy as np
plt.figure(figsize=(10,6))
plt.style.use('dark_background')

bars=dates.dates
height=dates.cases
y_pos = np.arange(len(bars))
plt.bar(y_pos, height)
plt.xticks(y_pos, bars)
plt.xticks(rotation=75)
plt.yticks(np.arange(0,40000,1500)) 
plt.xlabel("Date",color='magenta',size=30)
plt.ylabel("Confirmed Cases",color='magenta',size=30)
plt.title("How cases got increased from april 15 --May 2 in india ",color='red',size=30)
plt.show()
dates={'dates':['15/4/2020','16/4/2020','17/4/2020','18/4/2020','20/4/2020','21/4/2020','22/4/2020','23/4/2020','24/4/2020',
                '25/4/2020','26/4/2020','27/4/2020','28/4/2020','29/4/2020','30/4/2020','01/05/2020','02/05/2020'],
       'cases':[
apr_15['Cured/Discharged/Migrated'].sum(),
apr_16['Cured/Discharged/Migrated'].sum(),
apr_17['Cured/Discharged/Migrated'].sum(),
apr_19['Cured/Discharged/Migrated'].sum(),
apr_20['Cured/Discharged/Migrated'].sum(),
apr_21['Cured/Discharged/Migrated'].sum(),
apr_22['Cured/Discharged/Migrated'].sum(),
apr_23['Cured/Discharged/Migrated'].sum(),
apr_24['Cured/Discharged/Migrated'].sum(),
apr_25['Cured/Discharged/Migrated'].sum(),
apr_26['Cured/Discharged/Migrated'].sum(),
apr_27['Cured/Discharged/Migrated'].sum(),
apr_28['Cured/Discharged/Migrated'].sum(),
apr_29['Cured/Discharged/Migrated'].sum(),
apr_30['Cured/Discharged/Migrated'].sum(),
may_1['Cured/Discharged/Migrated'].sum(),
may_2['Cured/Discharged/Migrated'].sum()],
      }
cured=pd.DataFrame(data=dates,index=range(17))
cured.head()
import numpy as np
plt.figure(figsize=(10,6))
plt.style.use('dark_background')

bars=cured.dates
height=cured.cases
y_pos = np.arange(len(bars))
plt.bar(y_pos, height)
plt.xticks(y_pos, bars)
plt.xticks(rotation=75)
plt.yticks(np.arange(0,15000,2000)) 
plt.xlabel("Date",color='magenta',size=30)
plt.ylabel("Confirmed Cases",color='magenta',size=30)
plt.title(" Comparison of indian  Cured/Discharged/Migrated cases Covid-19 April 15th-May 2nd",color='red',size=30)
plt.show()
dates={'dates':['15/4/2020','16/4/2020','17/4/2020','18/4/2020','20/4/2020','21/4/2020','22/4/2020','23/4/2020','24/4/2020',
                '25/4/2020','26/4/2020','27/4/2020','28/4/2020','29/4/2020','30/4/2020','01/05/2020','02/05/2020'],
       'cases':[
apr_15['Death'].sum(),
apr_16['Death'].sum(),
apr_17['Death'].sum(),
apr_19['Death'].sum(),
apr_20['Death'].sum(),
apr_21['Death'].sum(),
apr_22['Death'].sum(),
apr_23['Death'].sum(),
apr_24['Death'].sum(),
apr_25['Death'].sum(),
apr_26['Death'].sum(),
apr_27['Death'].sum(),
apr_28['Death'].sum(),
apr_29['Death'].sum(),
apr_30['Death'].sum(),
may_1['Death'].sum(),
may_2['Death'].sum()],
      }
death=pd.DataFrame(data=dates,index=range(17))
death.tail()
import numpy as np
plt.figure(figsize=(10,6))
plt.style.use('dark_background')

bars=death.dates
height=death.cases
y_pos = np.arange(len(bars))
plt.bar(y_pos, height)
plt.xticks(y_pos, bars)
plt.xticks(rotation=75)
plt.yticks(np.arange(0,3000,500)) 
plt.xlabel("Date",color='magenta',size=30)
plt.ylabel("Confirmed Cases",color='magenta',size=30)
plt.title(" Comparison of indian Deaths Covid-19 April 15-May 2nd",color='blue',size=30)
plt.show()
plt.figure(figsize=(20,10),facecolor=(1,1,1))
height=dates['cases']
bars=dates['dates']
y_pos=np.arange(len(bars))


plt.plot(y_pos,height,'b-o',color='aqua')
plt.plot(y_pos,height,'r--',color='g',linewidth=4)
plt.xticks(y_pos,bars)
plt.xticks(rotation=90)
plt.title('Increase of cases from apr 15-may 2',size=40)
plt.ylabel('Cases per Day',size=30)
plt.xlabel('Date',size=30)
ax = plt.axes()
ax.set_facecolor("black")
ax.grid(False)
import seaborn as sns
may_2['active']=may_2['Total Confirmed cases (Including 111 foreign Nationals)'].sum()-(may_2['Cured/Discharged/Migrated']+may_2['Death'])
f,axes = plt.subplots(2, 2, figsize=(20,10))
sns.distplot( may_2["Cured/Discharged/Migrated"], color="b", ax=axes[0, 0])
sns.distplot( may_2["Death"], color="violet", ax=axes[0, 1])
sns.distplot( may_2['Total Confirmed cases (Including 111 foreign Nationals)'],color="olive", ax=axes[1, 0])
sns.distplot( may_2["active"], color="orange", ax=axes[1, 1])
f.subplots_adjust(hspace=.3,wspace=0.03) 




d=a-(b+c)
a=may_2['Total Confirmed cases (Including 111 foreign Nationals)'].sum()
b=may_2['Cured/Discharged/Migrated'].sum()
c=may_2['Death'].sum()

count=[a,b,c,d]
labels=['total confirmed','Cured','Death','Active']
plt.style.use('dark_background')

plt.figure(figsize=(10,6))
plt.bar(labels,count,color=['skyblue','salmon','green','yellow'])
plt.ylabel("Count", size=20)
plt.title("Comparioson of Cases,Covid-19", size=30)
plt.show()
print('Total confirmed cases:',a)
print('Total cured cases:',b)
print('Total death cases:',c)
print('Total Active cases:',d)
age=pd.read_csv('../input/covid19-in-india/AgeGroupDetails.csv')
age.info()
plt.figure(figsize=(10,10))
plt.title("percentage of age group affected from the data",fontsize=20)
#age['Agegroup'].as_tpye('int')
plt.pie(age['TotalCases'],colors = ['red','green','blue','red','violet','orange','indigo'],autopct="%1.1f%%")
plt.legend(age['AgeGroup'],loc='best')
plt.show() 
