# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import re

import seaborn as sns

from collections import Counter
from sklearn.linear_model import LinearRegression,LogisticRegression

from sklearn.tree import DecisionTreeRegressor

import warnings

warnings.filterwarnings('ignore')
agegrp=pd.read_csv('../input/covid19-in-india/AgeGroupDetails.csv')

covidindia=pd.read_csv('../input/covid19-in-india/covid_19_india.csv')

hospitalbeds=pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')

individualdetails=pd.read_csv('../input/covid19-in-india/IndividualDetails.csv')
agegrp.head()
agegrp.info()
hospitalbeds=hospitalbeds[:-2]

hospitalbeds.fillna(0,inplace=True)

hospitalbeds
hospitalbeds.info()
for col in hospitalbeds.columns[2:]:

    if hospitalbeds[col].dtype=='object':

        hospitalbeds[col]=hospitalbeds[col].astype('int64')
covidindia['Date']=pd.to_datetime(covidindia['Date'])
gender=individualdetails.gender

gender.dropna(inplace=True)

gender=gender.value_counts()

per=[]

for i in gender:

    perc=i/gender.sum()

    per.append(format(perc,'.2f'))

plt.figure(figsize=(10,6))    

plt.title('Case comparison according to gender',fontsize=20)

plt.pie(per,autopct='%1.1f%%')

plt.legend(gender.index,loc='best',title='Gender',fontsize=15)
perc=[]

for i in agegrp['Percentage']:

    per=float(re.findall("\d+\.\d+",i)[0])

    perc.append(per)

agegrp['Percentage']=perc

plt.figure(figsize=(20,10))

plt.title('Case percentage in the age group',fontsize=20)

plt.pie(agegrp['Percentage'],autopct='%1.1f%%')

plt.legend(agegrp['AgeGroup'],loc='best',title='Age Group')
plt.figure(figsize=(20,10))

plt.style.use('ggplot')

plt.title('Case comparison in different age group',fontsize=30)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Age Group',fontsize=20)

plt.ylabel('Confirmed Cases',fontsize=20)

plt.bar(agegrp['AgeGroup'],agegrp['TotalCases'],color=['Red','green','skyblue','orange','hotpink'],linewidth=3)

for i, j in enumerate(agegrp['TotalCases']):

    plt.text(i-.25, j,

              agegrp['TotalCases'][i], 

              fontsize=20 )
top=hospitalbeds.nlargest(20,'NumPrimaryHealthCenters_HMIS')



plt.figure(figsize=(20,20))

plt.title('Top States with number of Primary health centres',fontsize=30)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.xlabel('Number of Primary Health centers',fontsize=25)

plt.ylabel('States',fontsize=25)

plt.barh(top['State/UT'],top['NumPrimaryHealthCenters_HMIS'],color='blue',linewidth=1)
top=hospitalbeds.nlargest(20,'NumDistrictHospitals_HMIS')



plt.figure(figsize=(20,20))

plt.title('Top States with number of District Hospitals',fontsize=30)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.xlabel('Number of District Hospitals',fontsize=25)

plt.ylabel('States',fontsize=25)

plt.barh(top['State/UT'],top['NumDistrictHospitals_HMIS'],color='yellow',linewidth=3)
top=hospitalbeds.nlargest(20,'NumCommunityHealthCenters_HMIS')



plt.figure(figsize=(20,20))

plt.title('Top States with number of Community health centres',fontsize=30)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.xlabel('Number of Community Health centers',fontsize=25)

plt.ylabel('States',fontsize=25)

plt.barh(top['State/UT'],top['NumCommunityHealthCenters_HMIS'],color='green',linewidth=3)
top=hospitalbeds.nlargest(20,'NumRuralHospitals_NHP18')



plt.figure(figsize=(20,20))

plt.title('Top States with total number of Rural Hospitals',fontsize=30)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.xlabel('Number of Rural Hospitals',fontsize=25)

plt.ylabel('States',fontsize=25)

plt.barh(top['State/UT'],top['NumRuralHospitals_NHP18'],color='skyblue',linewidth=3)
top=hospitalbeds.nlargest(20,'NumUrbanHospitals_NHP18')



plt.figure(figsize=(20,20))

plt.title('Top States with total number of Urban Hospitals',fontsize=30)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.xlabel('Number of Urban Hospitals',fontsize=25)

plt.ylabel('States',fontsize=25)

plt.barh(top['State/UT'],top['NumUrbanHospitals_NHP18'],color='hotpink',linewidth=3)
df1=covidindia.groupby('Date')[['Cured','Deaths','Confirmed']].sum()
plt.figure(figsize=(20,10))

plt.style.use('ggplot')

plt.title('Observed Cases',fontsize=30)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.xlabel('Date',fontsize=20)

plt.ylabel('Number of cases',fontsize=20)

plt.plot(df1.index,df1['Confirmed'],linewidth=3,label='Confirmed',color='blue')

plt.plot(df1.index,df1['Cured'],linewidth=3,label='Cured',color='green')

plt.plot(df1.index,df1['Deaths'],linewidth=3,label='Deceased',color='red')

plt.legend(fontsize=20)
df2=covidindia.groupby('State/UnionTerritory')[['Cured','Deaths','Confirmed']].sum()
df2=df2.nlargest(20,'Confirmed')

plt.figure(figsize=(20,10))

plt.title('top states with confirmed cases',fontsize=30)

plt.xticks(rotation=90,fontsize=20)

plt.yticks(fontsize=20)

plt.xlabel('State',fontsize=20)

plt.ylabel('Cases',fontsize=20)

plt.plot(df2.index,df2.Confirmed,marker='o',mfc='black',label='Confirmed',markersize=10,linewidth=1,color='blue')

plt.plot(df2.index,df2.Deaths,marker='o',mfc='black',label='Deaths',markersize=10,linewidth=1,color='red')

plt.plot(df2.index,df2.Cured,marker='o',mfc='black',label='Cured',markersize=10,linewidth=1,color='green')

plt.legend(fontsize=20)
perc=[]

for i in df2.Confirmed:

    per=i/len(df2)

    perc.append(i)

plt.figure(figsize=(25,10))    

plt.title('Top states with confirmed cases with Percentage distribution ',fontsize=20)

plt.pie(perc,autopct='%1.1f%%')

plt.legend(df2.index,loc='upper right')
covidindia.isnull().sum()
covidindia["ConfirmedForeignNational"]=covidindia['ConfirmedForeignNational'].replace('-',0,inplace=True)

covidindia["ConfirmedIndianNational"]=covidindia['ConfirmedIndianNational'].replace('-',0,inplace=True)
covidindia.isnull().sum()
covidindia['ConfirmedIndianNational']=covidindia['ConfirmedIndianNational'].astype('float64')

covidindia['ConfirmedForeignNational']=covidindia['ConfirmedForeignNational'].astype('float64')
df3=covidindia.groupby('State/UnionTerritory')[['ConfirmedIndianNational','ConfirmedForeignNational']].sum()
df4=df3.nlargest(20,'ConfirmedIndianNational')

df5=df3.nlargest(20,'ConfirmedForeignNational')
plt.figure(figsize=(30,15))

plt.suptitle('Case comparison of indian national and foreign national',fontsize=40)

plt.subplot(121)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.barh(df4.index,df4.ConfirmedIndianNational,color='hotpink',linewidth=3)

plt.subplot(122)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.barh(df5.index,df5.ConfirmedForeignNational,color='cyan',linewidth=3)
plt.figure(figsize=(30,50))

plt.subplot(311)

plt.title('Confirmed Cases',fontsize=28)

plt.xticks(rotation=90,fontsize=20)

plt.yticks(fontsize=25)

plt.bar(df2.index,df2.Confirmed,color='red',linewidth=5)

plt.subplot(312)

plt.title('Cured Cases',fontsize=28)

plt.xticks(rotation=90,fontsize=20)

plt.yticks(fontsize=25)

plt.bar(df2.index,df2.Cured,color='green',linewidth=5)

plt.subplot(313)

plt.title('Deaths Cases',fontsize=28)

plt.xticks(rotation=90,fontsize=20)

plt.yticks(fontsize=25)

plt.bar(df2.index,df2.Deaths,color='blue',linewidth=5)
el=sns.catplot(x='State/UnionTerritory',y='Confirmed',kind='boxen',data=covidindia)

el.fig.set_figwidth(20)

el.fig.set_figheight(8)

el.set_xticklabels(rotation=90,fontsize=15)
el=sns.catplot(x='State/UnionTerritory',y='Cured',kind='boxen',data=covidindia)

el.fig.set_figwidth(20)

el.fig.set_figheight(8)

el.set_xticklabels(rotation=90,fontsize=15)
el=sns.catplot(x='State/UnionTerritory',y='Deaths',kind='boxen',data=covidindia)

el.fig.set_figwidth(20)

el.fig.set_figheight(8)

el.set_xticklabels(rotation=90,fontsize=15)
from sklearn.preprocessing import LabelEncoder

lbl=LabelEncoder()

covidindia['State/UnionTerritory']=lbl.fit_transform(covidindia['State/UnionTerritory'])
covidindia["ConfirmedForeignNational"]=covidindia['ConfirmedForeignNational'].fillna(0,inplace=False)

covidindia["ConfirmedIndianNational"]=covidindia['ConfirmedIndianNational'].fillna(0,inplace=False)
covidindia.isnull().sum()
covidindia['Date']=covidindia['Date'].astype('datetime64[ns]')
covidindia['date']=covidindia['Date'].dt.day

covidindia['month']=covidindia['Date'].dt.month
covidindia
linear=LinearRegression()

logistic=LogisticRegression()

tree=DecisionTreeRegressor()
from sklearn.model_selection import train_test_split

x=covidindia[['State/UnionTerritory','date','month','Cured','Deaths','ConfirmedIndianNational','ConfirmedForeignNational']]

y=covidindia['Confirmed']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
linear.fit(x_train,y_train)

logistic.fit(x_train,y_train)

tree.fit(x_train,y_train)
from sklearn.metrics import r2_score

prediction=logistic.predict(x_test)

score1=r2_score(y_test,prediction)
prediction=linear.predict(x_test)

score2=r2_score(y_test,prediction)
prediction=tree.predict(x_test)

score3=r2_score(y_test,prediction)
scores=[score1,score2,score3]

models=['LogisticRegression','LinearRegression','DecisionTreeRegressor']

plt.figure(figsize=(10,10))

plt.title('Accuracy comparison of Logistic Regression vs Linear Regression vs Decision tree models',fontsize=30)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.xlabel('models',fontsize=30)

plt.ylabel('Accuracy',fontsize=30)

plt.bar(models,scores,color=['blue','cyan','hotpink'],alpha=0.5,linewidth=4,edgecolor='black')

for i,v in enumerate(scores):

    plt.text(i-.15,v+.03,format(scores[i],'.2f'),fontsize=20)