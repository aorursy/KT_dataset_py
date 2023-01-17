import pandas as pd

import matplotlib.pyplot as plt

import re

from sklearn.linear_model import LinearRegression,LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeRegressor 

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
agegroup=pd.read_csv('../input/covid19-in-india/AgeGroupDetails.csv')

covid_19_india=pd.read_csv('../input/covid19-in-india/covid_19_india.csv')

hospital_beds=pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')

individual_details=pd.read_csv('../input/covid19-in-india/IndividualDetails.csv')
agegroup.head()
agegroup.info()
hospital_beds=hospital_beds[:-2]

hospital_beds.fillna(0,inplace=True)

hospital_beds
hospital_beds.info()

for col in hospital_beds.columns[2:]:

    if hospital_beds[col].dtype=='object':

        hospital_beds[col]=hospital_beds[col].astype('int64')
covid_19_india['Date']=pd.to_datetime(covid_19_india['Date'])

covid_19_india.head()
covid_19_india.info()
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
perc=[]

for i in agegroup['Percentage']:

    per=float(re.findall("\d+\.\d+",i)[0])

    perc.append(per)

agegroup['Percentage']=perc

plt.figure(figsize=(20,10))

plt.title('Percentage of cases in the age group',fontsize=20)

plt.pie(agegroup['Percentage'],autopct='%1.1f%%')

plt.legend(agegroup['AgeGroup'],loc='best',title='Age Group')
plt.figure(figsize=(20,10))

plt.style.use('ggplot')

plt.title('Comparing Total cases in different age group',fontsize=30)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Age Group',fontsize=20)

plt.ylabel('Confirmed Cases',fontsize=20)

plt.bar(agegroup['AgeGroup'],agegroup['TotalCases'],color=['darksalmon','plum','orange','blue','yellowgreen'],edgecolor='black',linewidth=3)

for i, v in enumerate(agegroup['TotalCases']):

    plt.text(i-.25, v,

              agegroup['TotalCases'][i], 

              fontsize=30 )


top_20=hospital_beds.nlargest(20,'NumPrimaryHealthCenters_HMIS')



plt.figure(figsize=(20,20))

plt.title('Top 20 States with number of Primary health centres',fontsize=30)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.xlabel('Number of Primary Health centers',fontsize=25)

plt.ylabel('States',fontsize=25)

plt.barh(top_20['State/UT'],top_20['NumPrimaryHealthCenters_HMIS'],color='red',edgecolor='black',linewidth=3)


top_20=hospital_beds.nlargest(20,'NumCommunityHealthCenters_HMIS')



plt.figure(figsize=(20,20))

plt.title('Top 20 States with number of Community health centres',fontsize=30)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.xlabel('Number of Community Health centers',fontsize=25)

plt.ylabel('States',fontsize=25)

plt.barh(top_20['State/UT'],top_20['NumCommunityHealthCenters_HMIS'],color='hotpink',edgecolor='black',linewidth=3)


top_20=hospital_beds.nlargest(20,'NumDistrictHospitals_HMIS')



plt.figure(figsize=(20,20))

plt.title('Top 20 States with number of District Hospitals',fontsize=30)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.xlabel('Number of District Hospitals',fontsize=25)

plt.ylabel('States',fontsize=25)

plt.barh(top_20['State/UT'],top_20['NumDistrictHospitals_HMIS'],color='lightgreen',edgecolor='black',linewidth=3)


top_20=hospital_beds.nlargest(20,'NumRuralHospitals_NHP18')



plt.figure(figsize=(20,20))

plt.title('Top 20 States with number of Rural Hospitals',fontsize=30)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.xlabel('Number of Rural Hospitals',fontsize=25)

plt.ylabel('States',fontsize=25)

plt.barh(top_20['State/UT'],top_20['NumRuralHospitals_NHP18'],color='blue',edgecolor='black',linewidth=3)


top_20=hospital_beds.nlargest(20,'NumUrbanHospitals_NHP18')



plt.figure(figsize=(20,20))

plt.title('Top 20 States with number of Urban Hospitals',fontsize=30)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.xlabel('Number of Urban Hospitals',fontsize=25)

plt.ylabel('States',fontsize=25)

plt.barh(top_20['State/UT'],top_20['NumUrbanHospitals_NHP18'],color='skyblue',edgecolor='black',linewidth=3)


top_20=hospital_beds.nlargest(20,'TotalPublicHealthFacilities_HMIS')

top_20=top_20[['State/UT','NumPrimaryHealthCenters_HMIS','NumCommunityHealthCenters_HMIS'      

                    ,'NumSubDistrictHospitals_HMIS','NumDistrictHospitals_HMIS'                 

                    ,'NumRuralHospitals_NHP18' ,'NumUrbanHospitals_NHP18']]

sns.pairplot(top_20,hue='State/UT')
df1=covid_19_india.groupby('Date')[['Cured','Deaths','Confirmed']].sum()
plt.figure(figsize=(20,10))

plt.style.use('ggplot')

plt.title('Observed Cases',fontsize=30)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.xlabel('Date',fontsize=20)

plt.ylabel('Number of cases',fontsize=20)

plt.plot(df1.index,df1['Confirmed'],linewidth=3,label='Confirmed',color='black')

plt.plot(df1.index,df1['Cured'],linewidth=3,label='Cured',color='green')

plt.plot(df1.index,df1['Deaths'],linewidth=3,label='Death',color='red')

plt.legend(fontsize=20)
df2=covid_19_india.groupby('State/UnionTerritory')[['Cured','Deaths','Confirmed']].sum()
df2=df2.nlargest(20,'Confirmed')

plt.figure(figsize=(20,10))

plt.title('top 20 states with confirmed cases',fontsize=30)

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

plt.barh(df4.index,df4.ConfirmedIndianNational,color='darkmagenta',edgecolor='black',linewidth=3)

plt.subplot(122)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.barh(df5.index,df5.ConfirmedForeignNational,color='lightskyblue',edgecolor='black',linewidth=3)
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
lbl=LabelEncoder()

covid_19_india['State/UnionTerritory']=lbl.fit_transform(covid_19_india['State/UnionTerritory'])

covid_19_india['date']=covid_19_india['Date'].dt.day

covid_19_india['month']=covid_19_india['Date'].dt.month
tree=DecisionTreeRegressor()

linear=LinearRegression()

logistic=LogisticRegression()

nb=GaussianNB()

forest=RandomForestClassifier()
x=covid_19_india[['State/UnionTerritory','date','month','Cured','Deaths','ConfirmedIndianNational','ConfirmedForeignNational']]

y=covid_19_india['Confirmed']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


tree.fit(x_train,y_train)

linear.fit(x_train,y_train)

logistic.fit(x_train,y_train)

nb.fit(x_train,y_train)

forest.fit(x_train,y_train)
from sklearn.metrics import r2_score

prediction=tree.predict(x_test)

score1=r2_score(y_test,prediction)
prediction=logistic.predict(x_test)

score2=r2_score(y_test,prediction)
prediction=linear.predict(x_test)

score3=r2_score(y_test,prediction)
prediction=forest.predict(x_test)

score4=r2_score(y_test,prediction)
prediction=nb.predict(x_test)

score5=r2_score(y_test,prediction)
scores=[score1,score2,score3,score4,score5]

models=['DecisionTreeRegressor','LogisticRegression','LinearRegression','RandomForestClassifier','GaussianNB']

plt.figure(figsize=(20,10))

plt.title('Comparing Accuracy of different models',fontsize=30)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.xlabel('models',fontsize=30)

plt.ylabel('Accuracy',fontsize=30)

plt.bar(models,scores,color=['red','magenta','cyan','blue','green'],alpha=0.5,linewidth=3,edgecolor='black')

for i,v in enumerate(scores):

    plt.text(i-.15,v+.03,format(scores[i],'.2f'),fontsize=20)