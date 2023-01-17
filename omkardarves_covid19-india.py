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

import matplotlib.pyplot as plt

import re

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor 

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import seaborn as sns
covid_19_india=pd.read_csv('../input/covid19-in-india/covid_19_india.csv')

individual_details=pd.read_csv('../input/covid19-in-india/IndividualDetails.csv')

population_india=pd.read_csv('../input/covid19-in-india/population_india_census2011.csv')

icmr_test=pd.read_csv('../input/covid19-in-india/ICMRTestingLabs.csv')

hospital_beds=pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')

population_india.head()
population_india.info()
hospital_beds=hospital_beds[:-2]

hospital_beds.fillna(0,inplace=True)

hospital_beds
hospital_beds.head()
hospital_beds.info()

for col in hospital_beds.columns[2:]:

    if hospital_beds[col].dtype=='object':

        hospital_beds[col]=hospital_beds[col].astype('int64')
covid_19_india.info()
covid_19_india['Date']=pd.to_datetime(covid_19_india['Date'])

covid_19_india.head()
individual_details.head()
from collections import Counter

gender=individual_details.gender

gender.dropna(inplace=True)

gender=gender.value_counts()

per=[]

for i in gender:

    perc=i/gender.sum()

    per.append(format(perc,'.2f'))

plt.figure(figsize=(10,6))    

plt.title('Comparing cases according to gender',fontsize=20)

plt.pie(per,autopct='%1.1f%%')

plt.legend(gender.index,loc='best',title='Gender',fontsize=15)
icmr_test.head()
top_10=hospital_beds.nlargest(10,'NumPrimaryHealthCenters_HMIS')



plt.figure(figsize=(10,10))

plt.title('Top 10 States with number of Primary health centres',fontsize=30)

plt.xticks(fontsize=10)

plt.yticks(fontsize=10)

plt.xlabel('Number of Primary Health centers',fontsize=15)

plt.ylabel('States',fontsize=15)

plt.barh(top_10['State/UT'],top_10['NumPrimaryHealthCenters_HMIS'],color='red',edgecolor='black',linewidth=3)
top_10=hospital_beds.nlargest(10,'NumCommunityHealthCenters_HMIS')



plt.figure(figsize=(10,10))

plt.title('Top 10 States with number of Community health centres',fontsize=15)

plt.xticks(fontsize=10)

plt.yticks(fontsize=10)

plt.xlabel('Number of Community Health centers',fontsize=15)

plt.ylabel('States',fontsize=15)

plt.barh(top_10['State/UT'],top_10['NumCommunityHealthCenters_HMIS'],color='hotpink',edgecolor='black',linewidth=3)
top_10=hospital_beds.nlargest(10,'NumDistrictHospitals_HMIS')



plt.figure(figsize=(10,10))

plt.title('Top 10 States with number of District Hospitals',fontsize=15)

plt.xticks(fontsize=10)

plt.yticks(fontsize=10)

plt.xlabel('Number of District Hospitals',fontsize=15)

plt.ylabel('States',fontsize=15)

plt.barh(top_10['State/UT'],top_10['NumDistrictHospitals_HMIS'],color='lightgreen',edgecolor='black',linewidth=3)
top_10=hospital_beds.nlargest(10,'NumRuralHospitals_NHP18')



plt.figure(figsize=(10,10))

plt.title('Top 10 States with number of Rural Hospitals',fontsize=15)

plt.xticks(fontsize=10)

plt.yticks(fontsize=10)

plt.xlabel('Number of Rural Hospitals',fontsize=15)

plt.ylabel('States',fontsize=15)

plt.barh(top_10['State/UT'],top_10['NumRuralHospitals_NHP18'],color='blue',edgecolor='black',linewidth=3)
top_10=hospital_beds.nlargest(10,'NumUrbanHospitals_NHP18')



plt.figure(figsize=(10,10))

plt.title('Top 10 States with number of Urban Hospitals',fontsize=15)

plt.xticks(fontsize=10)

plt.yticks(fontsize=10)

plt.xlabel('Number of Urban Hospitals',fontsize=15)

plt.ylabel('States',fontsize=15)

plt.barh(top_10['State/UT'],top_10['NumUrbanHospitals_NHP18'],color='skyblue',edgecolor='black',linewidth=3)
top_10=hospital_beds.nlargest(10,'TotalPublicHealthFacilities_HMIS')

top_10=top_10[['State/UT','NumPrimaryHealthCenters_HMIS','NumCommunityHealthCenters_HMIS'      

                    ,'NumSubDistrictHospitals_HMIS','NumDistrictHospitals_HMIS'                 

                    ,'NumRuralHospitals_NHP18' ,'NumUrbanHospitals_NHP18']]

sns.pairplot(top_10,hue='State/UT')
df1=covid_19_india.groupby('Date')[['Cured','Deaths','Confirmed']].sum()

df1.head()
plt.style.use('ggplot')

plt.title('Observed Cases',fontsize=30)

plt.xticks(fontsize=10)

plt.yticks(fontsize=10)

plt.xlabel('Date',fontsize=10)

plt.ylabel('Number of cases',fontsize=20)

plt.plot(df1.index,df1['Confirmed'],linewidth=3,label='Confirmed',color='black')

plt.plot(df1.index,df1['Cured'],linewidth=3,label='Cured',color='green')

plt.plot(df1.index,df1['Deaths'],linewidth=3,label='Death',color='red')

plt.legend(fontsize=10)
df2=covid_19_india.groupby('State/UnionTerritory')[['Cured','Deaths','Confirmed']].sum()

df2=df2.nlargest(10,'Confirmed')

plt.figure(figsize=(20,10))

plt.title('top 10 states with confirmed cases',fontsize=30)

plt.xticks(rotation=90,fontsize=20)

plt.yticks(fontsize=20)

plt.xlabel('State',fontsize=20)

plt.ylabel('Cases',fontsize=20)

plt.plot(df2.index,df2.Confirmed,marker='o',mfc='black',label='Confirmed',markersize=10,linewidth=5)

plt.plot(df2.index,df2.Deaths,marker='o',mfc='black',label='Deaths',markersize=10,linewidth=5)

plt.plot(df2.index,df2.Cured,marker='o',mfc='black',label='Cured',markersize=10,linewidth=5,color='green')

plt.legend(fontsize=20)
perc=[]

for i in df2.Confirmed:

    per=i/len(df2)

    perc.append(i)

plt.figure(figsize=(25,10))    

plt.title('Top 20 states with confirmed cases (Percentage distribution) ',fontsize=20)

plt.pie(perc,autopct='%1.1f%%')

plt.legend(df2.index,loc='upper right')
covid_19_india['ConfirmedForeignNational'].replace('-',0,inplace=True)

covid_19_india['ConfirmedIndianNational'].replace('-',0,inplace=True)

covid_19_india['ConfirmedIndianNational']=covid_19_india['ConfirmedIndianNational'].astype('int64')

covid_19_india['ConfirmedForeignNational']=covid_19_india['ConfirmedForeignNational'].astype('int64')
df3=covid_19_india.groupby('State/UnionTerritory')[['ConfirmedIndianNational','ConfirmedForeignNational']].sum()

df4=df3.nlargest(20,'ConfirmedIndianNational')

df5=df3.nlargest(20,'ConfirmedForeignNational')
plt.figure(figsize=(30,15))

plt.suptitle('Comparing cases of indian national and foreign national',fontsize=40)

plt.subplot(121)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.barh(df4.index,df4.ConfirmedIndianNational,color='magenta',edgecolor='black',linewidth=3)

plt.subplot(122)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.barh(df5.index,df5.ConfirmedForeignNational,color='skyblue',edgecolor='black',linewidth=3)
plt.figure(figsize=(30,40))

plt.subplot(311)

plt.title('Confirmed Cases',fontsize=30)

plt.xticks(rotation=90,fontsize=25)

plt.yticks(fontsize=25)

plt.bar(df2.index,df2.Confirmed,color='blue',linewidth=5,edgecolor='black')

plt.subplot(312)

plt.title('Cured Cases',fontsize=30)

plt.xticks(rotation=90,fontsize=25)

plt.yticks(fontsize=25)

plt.bar(df2.index,df2.Cured,color='green',linewidth=5,edgecolor='black')

plt.subplot(313)

plt.title('Deaths Cases',fontsize=30)

plt.xticks(rotation=90,fontsize=25)

plt.yticks(fontsize=25)

plt.bar(df2.index,df2.Deaths,color='red',linewidth=5,edgecolor='black')
df2=df2.nlargest(10,'Confirmed')

df2['state']=df2.index

sns.pairplot(df2,hue='state')
g=sns.catplot(x='State/UnionTerritory',y='Confirmed',kind='boxen',data=covid_19_india)

g.fig.set_figwidth(20)

g.fig.set_figheight(8)

g.set_xticklabels(rotation=90,fontsize=15)
g=sns.catplot(x='State/UnionTerritory',y='Cured',kind='boxen',data=covid_19_india)

g.fig.set_figwidth(20)

g.fig.set_figheight(8)

g.set_xticklabels(rotation=90,fontsize=15)
g=sns.catplot(x='State/UnionTerritory',y='Deaths',kind='boxen',data=covid_19_india)

g.fig.set_figwidth(20)

g.fig.set_figheight(8)

g.set_xticklabels(rotation=90,fontsize=15)
icmr_test.head()
from collections import Counter

type_of_lab=icmr_test.type

type_of_lab.dropna(inplace=True)

type_of_lab=type_of_lab.value_counts()

per=[]

for i in type_of_lab:

    perc=i/type_of_lab.sum()

    per.append(format(perc,'.2f'))

plt.figure(figsize=(10,6))    

plt.title('Comparing types of lab',fontsize=20)

plt.pie(per,autopct='%1.1f%%')

plt.legend(type_of_lab.index,loc='best',title='Type of Labs',fontsize=15,bbox_to_anchor=(1,1))
lbl=LabelEncoder()

covid_19_india['State/UnionTerritory']=lbl.fit_transform(covid_19_india['State/UnionTerritory'])

covid_19_india['date']=covid_19_india['Date'].dt.day

covid_19_india['month']=covid_19_india['Date'].dt.month
tree=DecisionTreeRegressor()

linear=LinearRegression()

forest=RandomForestClassifier()
x=covid_19_india[['State/UnionTerritory','date','month','Cured','Deaths','ConfirmedIndianNational','ConfirmedForeignNational']]

y=covid_19_india['Confirmed']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
tree.fit(x_train,y_train)

linear.fit(x_train,y_train)

forest.fit(x_train,y_train) #fit models
from sklearn.metrics import r2_score

prediction=tree.predict(x_test)

score1=r2_score(y_test,prediction)
prediction=linear.predict(x_test)

score2=r2_score(y_test,prediction)
prediction=forest.predict(x_test)

score3=r2_score(y_test,prediction)
scores=[score1,score2,score3]

scores