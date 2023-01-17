import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os
fread=pd.read_csv("../input/covidindia/AgeGroupDetails.csv")

fread1=fread.select_dtypes(include=['float64','int64'])

fread2=fread.select_dtypes(include=['object'])

fread.info()
fread1.head()
fread2.head()
from sklearn import preprocessing

le=preprocessing.LabelEncoder()

labels=le.fit_transform(fread['AgeGroup'])

print(len(le.classes_))

print(le.classes_)
sns.set(rc={'figure.figsize':(11,8)})

x=fread.AgeGroup

y=fread.TotalCases

y_pos=np.arange(len(x))

plt.xticks(y_pos,x)

plt.xticks(rotation=90)

plt.xlabel('Age Groups')





ax=sns.kdeplot(y_pos,y,cmap='Reds',shade=True,cbar=True)

covid=pd.read_csv('../input/covidindia/covid19.csv')
sns.pairplot(covid, palette="Set2")
plt.figure(figsize=(5,5))

cured=covid[covid['Cured']==True]

deaths=covid[covid['Deaths']==True]

slices_hours = [cured['Time'].count(),deaths['Time'].count()]

activities = ['Cured', 'Deaths']

colors = ['aqua', 'orange']

explode=(0,0.1)

plt.pie(slices_hours, labels=activities,explode=explode, colors=colors, startangle=90, autopct='%1.1f%%',shadow=True)

plt.show()





from sklearn.linear_model import LinearRegression

model=LinearRegression()



X=covid[['Cured']]

Y=covid[['Deaths']]

model.fit(X,Y)



Y_pred=model.predict(X)

plt.scatter(X,Y,color='green')

plt.plot(X,Y_pred,color='red')

plt.xlabel('cured')

plt.ylabel('deaths')



plt.show()
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(Y, Y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(Y, Y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y, Y_pred)))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y, Y_pred)))
covid['active']=covid['Confirmed']-(covid['Cured']+covid['Deaths'])

f,axes = plt.subplots(2, 2, figsize=(15,10))

sns.distplot( covid["Cured"] , color="blue", ax=axes[0, 0])

sns.distplot( covid["Deaths"] , color="violet", ax=axes[0, 1])

sns.distplot( covid["Confirmed"] , color="olive", ax=axes[1, 0])

sns.distplot( covid["active"] , color="orange", ax=axes[1, 1])

f.subplots_adjust(hspace=.3,wspace=0.03) 

symptoms={'symptoms':['Fever','Tiredness','Dry-cough','Shortness of breath','aches and pains','Sore throat','Diarrhoea','Nausea','vomiting','abdominal pain'],'percentage':[98.6,69.9,82,16.6,14.8,13.9,10.1,10.1,3.6,2.2]

    

}

symptoms=pd.DataFrame(data=symptoms,index=range(10))

symptoms

plt.figure(figsize=(10,5))

height=symptoms.percentage

bars=symptoms.symptoms

y_pos = np.arange(len(bars))



my_colors = ['red','green','blue','yellow','violet','orange','indigo']

plt.bar(y_pos, height,color=my_colors)

plt.xticks(y_pos, bars)

plt.xticks(rotation=90)

plt.xlabel("Symptoms", size=30)

plt.ylabel("Percentage", size=30)

plt.title("Symptoms of Covid-19", size=45)



plt.show()

plt.figure(figsize=(10,10))

plt.title("Symptoms of Corona",fontsize=20)

plt.pie(symptoms["percentage"],colors = ['red','green','blue','yellow','violet','orange','indigo'],autopct="%1.1f%%")

plt.legend(symptoms['symptoms'],loc='best')

plt.show() 
hosp=pd.read_csv("../input/covidindia/HospitalBedsIndia.csv")

hosp1=hosp.select_dtypes(include=['float64','int64'])

hosp2=hosp.select_dtypes(include=['object'])
health=hosp.drop([36,37])

obj=list(health.columns[2:8])



for ob in obj:

    health[ob]=health[ob].astype(int,errors='ignore')
plt.suptitle('HEALTH FACILITIES STATEWISE',fontsize=20)

fig = plt.figure(figsize=(20,10)) 

plt1 = fig.add_subplot(221) 

plt2 = fig.add_subplot(222) 

plt3 = fig.add_subplot(223) 

plt4 = fig.add_subplot(224) 



primary=health.nlargest(12,'NumPrimaryHealthCenters_HMIS')



plt1.set_title('Primary Health Centers')

plt1.barh(primary['State/UT'],primary['NumPrimaryHealthCenters_HMIS'],color ='gold');



community=health.nlargest(12,'NumCommunityHealthCenters_HMIS')

plt2.set_title('Community Health Centers')

plt2.barh(community['State/UT'],community['NumCommunityHealthCenters_HMIS'],color='coral')



dist=health.nlargest(12,'NumDistrictHospitals_HMIS')

plt3.set_title("District Hospitals")

plt3.barh(dist['State/UT'],dist['NumDistrictHospitals_HMIS'],color='lightskyblue')



subd=health.nlargest(12,'TotalPublicHealthFacilities_HMIS')

plt4.set_title('PUblic Health Facilities')

plt4.barh(subd['State/UT'],subd['TotalPublicHealthFacilities_HMIS'],color='violet')



fig.subplots_adjust(hspace=.5,wspace=0.2) 

indiv=pd.read_csv("../input/covidindia/IndividualDetails.csv")

indiv2=indiv.select_dtypes(include=['float64','int64'])

indiv3=indiv.select_dtypes(include=['object'])

plt.figure(figsize=(5,10))

male=indiv[indiv['gender']=='M']

female=indiv[indiv['gender']=='F']

slices_hours = [male['age'].count(),female['age'].count()]

activities = ['Male', 'Female']

colors = ['green', 'gold']

explode=(0,0.1)

plt.pie(slices_hours, labels=activities,explode=explode, colors=colors, startangle=180, autopct='%1.1f%%',shadow=True)

plt.show()

april=pd.read_csv("../input/april2020/2020_04_08.csv")

april2=april.select_dtypes(include=['float64','int64'])

april3=april.select_dtypes(include=['object'])

from sklearn import preprocessing

le=preprocessing.LabelEncoder()

labels=le.fit_transform(april['Death'])

print(len(le.classes_))

print(le.classes_)
from sklearn import preprocessing

le=preprocessing.LabelEncoder()

labels=le.fit_transform(april['Cured/Discharged/Migrated'])

print(len(le.classes_))

print(le.classes_)
april.Death.value_counts().plot.bar(color=['gold','coral','aqua','skyblue','pink','violet'])

cases=april['Total cases'].sum()

cdm=april['Cured/Discharged/Migrated'].sum()

d=april['Death'].sum()



plt.figure(figsize=(5,5))

plt.title("Current situartion in india",fontsize=20)

labels='Total Cases','Cured','Death'

sizes=[cases,cdm,d]

explode=[0.1,0.1,0.1]

colors=['gold','yellowgreen','aqua']

plt.pie(sizes,labels=labels,colors=colors,explode=explode,autopct='%1.1f%%',shadow=True,startangle=90)

plt.show() 
april['active']=april['Total cases']-april['Death']-april['Cured/Discharged/Migrated']

print(april['active'].sum())

print(april['Total cases'].sum())
cases=april['active'].sum()

cdm=april['Cured/Discharged/Migrated'].sum()

d=april['Death'].sum()



plt.figure(figsize=(7,7))

plt.title("Current situartion in india",fontsize=20)

labels='Total Cases','Cured','Death'

sizes=[cases,cdm,d]

explode=[0.1,0.1,0.1]

colors=['lightcoral','yellowgreen','skyblue']

plt.axis('equal')

plt.pie(sizes,labels=labels,colors=colors,explode=explode,autopct='%1.1f%%',shadow=True,startangle=90)

plt.legend(labels, loc="best")

plt.show() 
april['mortality']=april['Death']/april['active']*100

print(april['mortality'])
plt.figure(figsize=(9, 10))



height=april['Death']

bars=april['Name of State / UT']

y_pos=np.arange(len(bars))

plt.barh(y_pos,height,color=['pink','lightcoral','violet','gold','lightskyblue'])

plt.yticks(y_pos,bars)

plt.title('Deaths in States',size=30)

plt.ylabel('States',size=20)

plt.xlabel('Deaths',size=20)

plt.show()
plt.figure(figsize=(9, 10))



height=april['Cured/Discharged/Migrated']

bars=april['Name of State / UT']

y_pos=np.arange(len(bars))

plt.barh(y_pos,height,color=['pink','lightcoral','violet','gold','lightskyblue'])

plt.yticks(y_pos,bars)

plt.title('Cured/Discharged/Migrated in States',size=30)

plt.ylabel('States',size=20)

plt.xlabel('Cured',size=20)

plt.show()
plt.figure(figsize=(9, 10))



height=april['mortality']

bars=april['Name of State / UT']

y_pos=np.arange(len(bars))

plt.barh(y_pos,height,color=['pink','lightcoral','violet','gold','lightskyblue'])

plt.yticks(y_pos,bars)

plt.title('Mortality Rate according to States',size=30)

plt.ylabel('States',size=20)

plt.xlabel('Mortality Rate',size=20)

plt.show()
plt.figure(figsize=(9, 10))



height=april['active']

bars=april['Name of State / UT']

y_pos=np.arange(len(bars))

plt.barh(y_pos,height,color=['pink','lightcoral','violet','gold','lightskyblue'])

plt.yticks(y_pos,bars)

plt.title('State-wise active cases',size=30)

plt.ylabel('States',size=20)

plt.xlabel('active',size=20)

plt.show()
bara=pd.read_csv("../input/12th-april/2020_04_12.csv")

bara2=bara.select_dtypes(include=['float64','int64','object'])



bara2.head(10)
cases=bara['Total cases'].sum()

print(cases)
last=(april['Total cases'].sum())

increase=cases-last

print(increase)
percent=increase/last*100

print(percent)
dates={'dates':['12/4/2020','8/4/2020'],'cases':[cases,last]}

dates=pd.DataFrame(data=dates,index=range(2))

dates





plt.figure(figsize=(5,5))

bars=dates.dates

height=dates.cases

y_pos = np.arange(len(bars))

plt.bar(y_pos, height,color=['skyblue','salmon'])

plt.xticks(y_pos, bars)

plt.xticks(rotation=90)

plt.xlabel("Date", size=20)

plt.ylabel("Cases", size=20)

plt.title("Comparioson of Cases,Covid-19", size=30)





plt.show()
plt.figure(figsize=(9, 10))



height=bara['Total cases']

bars=april['Name of State / UT']

y_pos=np.arange(len(bars))



plt.barh(y_pos,height,color=['pink','lightcoral','violet','gold','lightskyblue'])

plt.yticks(y_pos,bars)

plt.title('Total cases in States',size=30)

plt.ylabel('States',size=20)

plt.xlabel('Cases',size=20)

plt.show()
perday=pd.read_csv('../input/12th-april/perday_new_cases.csv')

perday2=perday.select_dtypes(include=['int64','float64','object'])

perday.info()
plt.figure(figsize=(20,10),facecolor=(1,1,1))

height=perday['New Daily Cases']

bars=perday['Date']

y_pos=np.arange(len(bars))





plt.plot(y_pos,height,'b-o',color='aqua')

plt.plot(y_pos,height,'r--',color='orange',linewidth=4)

plt.xticks(y_pos,bars)

plt.xticks(rotation=90)

plt.title('New Daily Cases',size=40)

plt.ylabel('Cases per Day',size=30)

plt.xlabel('Date',size=30)

ax = plt.axes()

ax.set_facecolor("black")

ax.grid(False)





lifestyle={'lifestyle':['Not waste food','Be environment conscious','Be more mindful of Health','Become more hygienic','More Family Time','Spend less on Clothes','Made in India products','Take work more seriously','Boycott Chinese goods'],

          'percentage':[67.7,45.6,44.3,40.5,31.8,31.4,26.4,25.5,24.6]}

lifestyle=pd.DataFrame(data=lifestyle,index=range(9))

lifestyle
plt.figure(figsize=(10,10))

plt.title("Impact on lifestyle of Indians",fontsize=20)

plt.pie(lifestyle["percentage"],colors = ['red','gold','green','blue','purple','violet','orange','indigo','coral'],autopct="%1.1f%%",shadow=True)

plt.legend(lifestyle['lifestyle'],loc='upper right')

plt.show() 
ap=pd.read_csv("../input/16april/16.csv")

app=pd.read_csv("../input/covid19-corona-virus-india-dataset/complete.csv")

ap1=ap.select_dtypes(include=['float64','object','int64'])

ap.info()
ap1.head()
deaths=ap1['Death'].sum()

print(deaths)
cure=ap1['Cured/Discharged/Migrated'].sum()

print(cure)
plt.figure(figsize=(5,5))

ap1['Cured/Discharged/Migrated'].hist(color='pink',bins=50)
plt.figure(figsize=(5,5))

ap1['Death'].hist(color='violet',bins=50)
total=ap1['Total Confirmed cases'].sum()

print(total)
kr=(round(deaths/total*100,2));

print("Currently the Mortality Rate of India is:",kr)
heatmap1_data = pd.pivot_table(app, values='Death', 

                     index=['Name of State / UT'], 

                     columns='Date')





sns.heatmap(heatmap1_data, cmap="RdYlGn",linewidths=0.01)
h=pd.pivot_table(app,values='Total Confirmed cases',

index=['Name of State / UT'],

columns='Date')



sns.heatmap(h, cmap=['skyblue','salmon','gold','green'],linewidths=0.05)
import folium





m = folium.Map(location=[20.5937, 78.9629],zoom_start=5)







for lat,lon,area,count in zip(ap1['Latitude'],ap1['Longitude'],ap1['Name of State / UT'],ap1['Total Confirmed cases']):

     folium.CircleMarker([lat, lon],

                            popup=area,

                            radius=count*0.02,

                            color='red',

                            fill=True,

                            fill_opacity=0.7,

                            fill_color='salmon',

                           ).add_to(m)

m.save('LA collisions.html')

m
import folium





m = folium.Map(location=[20.5937, 78.9629],zoom_start=5)







for lat,lon,area,count in zip(ap1['Latitude'],ap1['Longitude'],ap1['Name of State / UT'],ap1['Death']):

     folium.CircleMarker([lat, lon],

                            popup=area,

                            radius=count*0.04,

                            color='purple',

                            fill=True,

                            fill_opacity=0.7,

                            fill_color='violet',

                           ).add_to(m)

m.save('LA collisions.html')

m
latest=pd.read_csv('../input/dynamiccovid19india-statewise/26-04-2020.csv')

latest.info()

latest.select_dtypes(include=['object','int64','float64'])
print("Total number of Cases")

latest['Total Confirmed cases (Including 111 foreign Nationals)'].sum()
print("Total number of Deaths")

latest['Death'].sum()
import folium





m = folium.Map(location=[20.5937, 78.9629],zoom_start=5)



latest=latest.drop([32,33,34])



for lat,lon,area,count in zip(ap1['Latitude'],ap1['Longitude'],ap1['Name of State / UT'],latest['Total Confirmed cases (Including 111 foreign Nationals)']):

     folium.CircleMarker([lat, lon],

                            popup=area,

                            radius=count*0.01,

                            color='olive',

                            fill=True,

                            fill_opacity=0.7,

                            fill_color='yellow',

                           ).add_to(m)

m.save('LA collisions.html')

m


sns.catplot(y="Name of State / UT", x="Total Confirmed cases (Including 111 foreign Nationals)",height=15,aspect=1,kind="bar", data=latest)

plt.title('Total confirmed Cases',size=30)

plt.show()

cases=latest['Total Confirmed cases (Including 111 foreign Nationals)'].sum()

cdm=latest['Cured/Discharged/Migrated'].sum()

d=latest['Death'].sum()



plt.figure(figsize=(5,5))

plt.title("Current situartion in india",fontsize=20)

labels='Total Cases','Cured','Death'

sizes=[cases,cdm,d]

explode=[0.1,0.1,0.1]

colors=['gold','yellowgreen','aqua']

plt.pie(sizes,labels=labels,colors=colors,explode=explode,autopct='%1.1f%%',shadow=True,startangle=90)

plt.show() 
plt.figure(figsize=(20,20))

plt.barh(latest["Name of State / UT"],latest['Total Confirmed cases (Including 111 foreign Nationals)'],label="Confirm Cases",color='gold')

plt.barh(latest["Name of State / UT"], latest['Cured/Discharged/Migrated'],label="Recovered Cases",color='coral')

plt.xlabel('Cases',size=30)

plt.ylabel("States",size=30)

plt.legend(frameon=True, fontsize=12)

plt.title('Recoveries and Total Number of Cases Statewise',fontsize = 20)

plt.show()
plt.figure(figsize=(20,10))

plt.bar(latest["Name of State / UT"], latest['Cured/Discharged/Migrated'],label="Recovered Cases",color='coral')

plt.bar(latest["Name of State / UT"],latest['Death'],label="Death",color='skyblue')

plt.xlabel('states',size=30)

plt.xticks(rotation=90)

plt.ylabel("cases",size=30)

plt.legend(frameon=True, fontsize=12)

plt.title('Recoveries and Death Cases Statewise',fontsize = 20)

plt.show()
rec=pd.read_csv('../input/yolooo/30-04-2020.csv')

rec.head()
rec['Total Confirmed cases (Including 111 foreign Nationals)'].argmax()
rec.loc[18]
rec['Death'].argmax()
rec.loc[18]
rec['Cured/Discharged/Migrated'].argmax()
rec['Cured/Discharged/Migrated'].argmin()
rec.loc[20]
rec['Total Confirmed cases (Including 111 foreign Nationals)'].argmin()
rec.loc[2]
deaths={'states':rec["Name of State / UT"] ,'deaths':rec['Death']==0}

df=pd.DataFrame(deaths,index=range(32))

df
import folium

from folium import plugins





m = folium.Map(location=[20.5937, 78.9629],zoom_start=5,tiles='cartodbpositron')



for lat,lon,area,count in zip(rec['Latitude'],rec['Longitude'],rec['Name of State / UT'],rec['Total Confirmed cases (Including 111 foreign Nationals)']):

    folium.CircleMarker([lat, lon],

                            popup=area,

                            radius=count*0.005,

                            color='neon',

                            fill=True,

                            fill_opacity=0.7,

                            fill_color='skyblue',

                           ).add_to(m)

location_data = rec[['Latitude', 'Longitude']].as_matrix()



# plot heatmap

m.add_child(plugins.HeatMap(location_data, radius=40,blur=10))
d=pd.read_csv("../input/1may30april/may.csv")

d.head()
d.groupby("State/UnionTerritory")['Confirmed'].mean()


d.groupby("State/UnionTerritory")['Confirmed'].mean().plot(kind='barh',color='skyblue',figsize=(10,15))

d.groupby("State/UnionTerritory")['Cured'].mean().plot(kind='barh',color='coral')



may=pd.read_csv('../input/dynamiccovid19india-statewise/11-05-2020.csv')

may.head()
may=may.drop([33,34,35,36,37])
cases=may['Total Confirmed cases*'].sum()

cases
prev=rec['Total Confirmed cases (Including 111 foreign Nationals)'].sum()

print("The increase in the number of cases in the past 10 days",cases-prev)
percentage=(cases-prev)/prev*100

print("The increse in percentage of cases from 30th April to 11th May 2020:",percentage)
df=may.nlargest(5,'Total Confirmed cases*')
sns.barplot( x=df["Name of State / UT"], y=df["Total Confirmed cases*"], palette="Blues")

plt.title("TOP 5 STATES WITH THE MAXIMUM NUMBER OF CASES",size=25)

sns.barplot( x=df["Name of State / UT"], y=df["Cured/Discharged/Migrated"], palette="Greens")

plt.title("TOP 5 Infected STATES and the number of people cured",size=25)
sns.barplot( x=df["Name of State / UT"], y=df["Deaths**"], palette="Reds")

plt.title("TOP 5 Infected STATES and the number of Deaths",size=25)
sns.pairplot(may, kind="scatter")



 
