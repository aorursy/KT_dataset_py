import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os
Age=pd.read_csv("../input/covid19-in-india/AgeGroupDetails.csv")

Age1=Age.select_dtypes(include=['float64','int64'])

Age2=Age.select_dtypes(include=['object'])

Age.info()
Age2.head()
from sklearn import preprocessing

le=preprocessing.LabelEncoder()

labels=le.fit_transform(Age['AgeGroup'])

print(len(le.classes_))

print(le.classes_)
sns.set(rc={'figure.figsize':(11,8)})

x=Age.AgeGroup

y=Age.TotalCases

y_pos=np.arange(len(x))

plt.xticks(y_pos,x)

plt.xticks(rotation=90)

plt.xlabel('Age Groups')





ax=sns.kdeplot(y_pos,y,cmap='Blues',shade=True,cbar=True)
corona=pd.read_csv('../input/covid19-in-india/covid_19_india.csv')
sns.pairplot(corona, palette="Set2")
plt.figure(figsize=(5,5))

cured=corona[corona['Cured']==True]

deaths=corona[corona['Deaths']==True]

slices_hours = [cured['Time'].count(),deaths['Time'].count()]

activities = ['Cured', 'Deaths']

colors = ['red', 'orange']

explode=(0,0.1)

plt.pie(slices_hours, labels=activities,explode=explode, colors=colors, startangle=90, autopct='%1.1f%%',shadow=True)

plt.show()
from sklearn.linear_model import LinearRegression

model=LinearRegression()



X=corona[['Cured']]

Y=corona[['Deaths']]

model.fit(X,Y)



Y_pred=model.predict(X)

plt.scatter(X,Y,color='green')

plt.plot(X,Y_pred,color='blue')

plt.xlabel('cured')

plt.ylabel('deaths')



plt.show()
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(Y, Y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(Y, Y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y, Y_pred)))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y, Y_pred)))
symptoms={'symptoms':['Fever','Tiredness','Dry-cough','Shortness of breath','aches and pains','Sore throat','Diarrhoea','Nausea','vomiting','abdominal pain'],'percentage':[98.6,69.9,82,16.6,14.8,13.9,10.1,10.1,3.6,2.2]

    

}

symptoms=pd.DataFrame(data=symptoms,index=range(10))

symptoms
plt.figure(figsize=(10,5))

height=symptoms.percentage

bars=symptoms.symptoms

y_pos = np.arange(len(bars))



my_colors = ['orange','green','pink','yellow','blue','red','indigo']

plt.bar(y_pos, height,color=my_colors)

plt.xticks(y_pos, bars)

plt.xticks(rotation=90)

plt.xlabel("Symptoms", size=30)

plt.ylabel("Percentage", size=30)

plt.title("Symptoms of COVID-19", size=45)



plt.show()
plt.figure(figsize=(10,10))

plt.title("Symptoms of Corona",fontsize=20)

plt.pie(symptoms["percentage"],colors = ['red','green','blue','yellow','violet','orange','indigo'],autopct="%1.1f%%")

plt.legend(symptoms['symptoms'],loc='best')

plt.show() 
details=pd.read_csv("../input/covidindia/IndividualDetails.csv")

details1=details.select_dtypes(include=['float64','int64'])

details2=details.select_dtypes(include=['object'])
plt.figure(figsize=(5,10))

male=details[details['gender']=='M']

female=details[details['gender']=='F']

slices_hours = [male['age'].count(),female['age'].count()]

activities = ['Male', 'Female']

colors = ['red', 'orange']

explode=(0,0.1)

plt.pie(slices_hours, labels=activities,explode=explode, colors=colors, startangle=180, autopct='%1.1f%%',shadow=True)

plt.show()
april=pd.read_csv("../input/dynamiccovid19india-statewise/17-04-2020.csv")

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
april.Death.value_counts().plot.bar(color=['red','yellow','blue','orange','pink','violet'])

cases=april['Total Confirmed cases (Including 76 foreign Nationals)'].sum()

cdm=april['Cured/Discharged/Migrated'].sum()

d=april['Death'].sum()



plt.figure(figsize=(5,5))

plt.title("Current situartion in india",fontsize=20)

labels='Total Confirmed cases (Including 76 foreign Nationals)','Cured','Death'

sizes=[cases,cdm,d]

explode=[0.1,0.1,0.1]

colors=['blue','green','gold']

plt.pie(sizes,labels=labels,colors=colors,explode=explode,autopct='%1.1f%%',shadow=True,startangle=90)

plt.show() 
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
plt.figure(figsize=(9, 10))



height=april['Death']

bars=april['Name of State / UT']

y_pos=np.arange(len(bars))

plt.barh(y_pos,height,color=['red','lightgreen','blue','green','lightskyblue'])

plt.yticks(y_pos,bars)

plt.title('Deaths in States',size=30)

plt.ylabel('States',size=20)

plt.xlabel('Deaths',size=20)

plt.show()