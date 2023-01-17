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
#import libraries



import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

import matplotlib.patches as mpatches

from sklearn import  linear_model

import numpy as np

import sklearn

import time

import datetime

import operator

import random

from sklearn.svm import  SVR

from sklearn.metrics import  mean_squared_error,mean_absolute_error,median_absolute_error

import math

import seaborn as sns

from plotly.offline import init_notebook_mode, iplot

from sklearn.model_selection import  RandomizedSearchCV,train_test_split

import plotly.express as px

import plotly.graph_objs as go

from sklearn    import metrics, svm

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn import  linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

confirmed_cases=pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

deaths_cases=pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

recovered_cases=pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

turkey=pd.read_csv("https://raw.githubusercontent.com/ozanerturk/covid19-turkey-api/master/dataset/timeline.csv")

turkey_cities=pd.read_csv("/kaggle/input/number-of-cases-in-the-city-covid19-turkey/number_of_cases_in_the_city.csv")
turkey.head()
plt.figure(figsize=(25,10))

sns.barplot(x=turkey['date'], y=turkey['cases'])

plt.xticks(rotation= 90)

plt.xlabel('Confirmed Cases')

plt.title('Confirmed Cases for every day (TURKEY)',fontsize = 50,color='black')

plt.ylabel('date (DAYS)')

plt.grid()
f,ax1 = plt.subplots(figsize =(25,10))



sns.pointplot(x=turkey['date'],y=turkey['cases'],color='green',alpha=0.5)

plt.xlabel('date (Days)',fontsize = 10,color='black')

plt.ylabel('Number of Cases',fontsize = 10,color='black')

plt.xticks(rotation= 90)

plt.title('Curve for Number of Cases Over Time (TURKEY)',fontsize = 40,color='black')

plt.grid()
plt.figure(figsize=(25,10))

sns.barplot(x=turkey['date'][7:], y=turkey['tests'][7:])

plt.xticks(rotation= 90)

plt.ylabel('Number of tests')

plt.title('Tests for every day (TURKEY)',fontsize = 50,color='black')

plt.xlabel('date (DAYS)')

plt.grid()
#Number of tests over Time in turkey



f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x=turkey['date'][7:],y=turkey['tests'][7:],color='blue',alpha=0.5)

plt.xlabel('date (Days)',fontsize = 10,color='black')

plt.ylabel('Number of testS',fontsize = 10,color='black')



plt.xticks(rotation= 90)

plt.title('Curve for Number of Tests Over Time (TURKEY)',fontsize = 40,color='black')

plt.grid()
plt.figure(figsize=(20,10))

sns.barplot(x=turkey['date'], y=turkey['deaths'])

plt.xticks(rotation= 90)

plt.ylabel('Number of deaths')

plt.title('Deaths for every day (TURKEY)',fontsize = 50,color='black')

plt.xlabel('date (DAYS)')

plt.grid()
f,ax1 = plt.subplots(figsize =(20,10))



sns.pointplot(x=turkey['date'],y=turkey['deaths'],color='red',alpha=0.5)

plt.xlabel('date (Days)',fontsize = 10,color='black')

plt.ylabel('Number of test',fontsize = 10,color='black')

plt.xticks(rotation= 90)

plt.title('Curve for Number of Deaths Over Time (TURKEY)',fontsize = 40,color='black')

plt.grid()
plt.figure(figsize=(20,10))

sns.barplot(x=turkey['date'], y=turkey['recovered'])

plt.xticks(rotation= 90)

plt.ylabel('Number of Recovered')

plt.title('Recovered for every day (TURKEY)',fontsize = 50,color='black')

plt.xlabel('date (DAYS)')

plt.grid()
f,ax1 = plt.subplots(figsize =(20,10))



sns.pointplot(x=turkey['date'],y=turkey['recovered'],color='orange',alpha=0.5)

plt.xlabel('date (Days)',fontsize = 30,color='black')

plt.ylabel('Number of Recovered',fontsize = 30,color='black')

plt.xticks(rotation= 90)

plt.title('Curve for Number of Recovered Cases Over Time (TURKEY)',fontsize = 40,color='black')

plt.grid()
# visualize Tests VS Confirmed 

f,ax1 = plt.subplots(figsize =(25,12))

sns.pointplot(x=turkey['tests'],y=turkey['deaths'],color='green',alpha=0.5)





plt.xlabel('Number Of Tests',fontsize = 30,color='black')

plt.ylabel('Number of Deaths ',fontsize = 30,color='black')

deaths = mpatches.Patch(color='green', label='Deaths')



plt.legend(handles=[deaths])

plt.xticks(rotation= 90)

plt.title(' Deaths  Cases / Tests Number (TURKEY)',fontsize = 40,color='black')

plt.grid()
# visualize Tests VS Confirmed 

f,ax1 = plt.subplots(figsize =(25,12))



sns.pointplot(x=turkey['tests'],y=turkey['cases'],color='red',alpha=0.5)





plt.xlabel('Number Of Tests',fontsize = 30,color='black')

plt.ylabel('Number of  confirmed',fontsize = 30,color='black')



Confirmed = mpatches.Patch(color='red', label='Confirmed')



plt.legend(handles=[Confirmed])

plt.xticks(rotation= 90)

plt.title(' Confirmed Cases / Tests Number (TURKEY)',fontsize = 40,color='black')

plt.grid()
# visualize Recovered VS Confirmed VS Deaths

f,ax1 = plt.subplots(figsize =(25,12))

sns.pointplot(x=turkey['date'],y=turkey['recovered'],color='green',alpha=0.5)

sns.pointplot(x=turkey['date'],y=turkey['cases'],color='red',alpha=0.5)

sns.pointplot(x=turkey['date'],y=turkey['deaths'],color='blue',alpha=0.5)

plt.xlabel('Date (Days)',fontsize = 30,color='black')

plt.ylabel('Number of Confirmed / Recovered / Deaths',fontsize = 30,color='black')

recovered = mpatches.Patch(color='green', label='Recovered')

Confirmed = mpatches.Patch(color='red', label='Confirmed')

Deaths = mpatches.Patch(color='blue', label='Deaths')



plt.legend(handles=[recovered,Confirmed,Deaths])

plt.xticks(rotation= 90)

plt.title('Recovered  VS Confirmed VS Deaths Cases Over Time (TURKEY)',fontsize = 40,color='black')

plt.grid()


plt.figure(figsize=(30,15))

plt.subplot(2,2,1)

plt.plot(range(len(turkey['date'])), np.log10(turkey['deaths']))

plt.title('Log of Coronavirus Deaths Over Time (TURKEY)',fontsize = 30,color='black')

plt.ylabel('Number of Deaths',size=20)

plt.grid()

plt.subplot(2,2,2)

plt.plot(range(len(turkey['date'])), np.log10(turkey['cases']))

plt.title('Log of Coronavirus Confirmed Cases Over Time (TURKEY)',fontsize = 30,color='black')

plt.ylabel('Number of Confirmed',size=20)

plt.grid()

plt.subplot(2,2,3)

plt.plot(range(len(turkey['date'])), np.log10(turkey['recovered']))

plt.title('Log of Coronavirus Recovered Over Time (TURKEY)',fontsize = 30,color='black',)

plt.ylabel('Number of Recovered',size=20)

plt.grid()

plt.figure(figsize=(20,10))

plt.subplot(2,2,1)

turkey['deaths'].plot.box()

plt.subplot(2,2,2)

turkey['cases'].plot.box()

plt.subplot(2,2,3)

turkey['recovered'].plot.box()

plt.subplot(2,2,4)

turkey['deaths'].plot.box()
# visual Comparation amoung Recovered , Confirmed , Deaths and Test Numbers

plt.figure(figsize=(40,20))

plt.subplot(2,2,1)

plt.plot(turkey['deaths'],color='green',alpha=0.5, marker='o',lw=4.0, linestyle='dashed')

plt.title('Deaths Cases (TURKEY)',fontsize = 30,color='black')

plt.ylabel('Number of Deaths',size=20)

plt.grid()

plt.subplot(2,2,2)

plt.plot(turkey['cases'],color='red',alpha=0.5, marker='o', lw=4.0,linestyle='dashed')

plt.title('Confirmed Cases (TURKEY)',fontsize = 30,color='black')

plt.ylabel('Number of confirmed',size=20)

plt.grid()

plt.subplot(2,2,3)

plt.plot(turkey['recovered'],color='blue',alpha=0.5, marker='o', lw=4.0,linestyle='dashed')

plt.title('Recoverd Cases (TURKEY)',fontsize = 30,color='black',)

plt.ylabel('Number of recovered',size=20)

plt.grid()

plt.subplot(2,2,4)

plt.plot(turkey['tests'],color='brown',alpha=0.5, marker='o', lw=4.0, linestyle='dashed')

plt.title('Tests Number (TURKEY)',fontsize = 30,color='black')

plt.ylabel('Number of tests',size=20)

plt.grid()
turkey.corr()


f,ax = plt.subplots(figsize=(15, 10))

sns.heatmap(turkey.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.show()
turkey_cities.head()
turkey_cities.sort_values(by=['Number of Case'], ascending=False, inplace = True)

fig = px.pie(turkey_cities.head(10), values='Number of Case', names='Province', title='Top 10 provinces with cases')

fig.show()
province_df2 = turkey_cities

plt.figure(figsize=(20,10))

fig = px.pie(province_df2[1:], values='Number of Case', names='Province', title='Number of death cases in every province except istabul', 

             hover_data=['Number of Case'])

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
plt.figure(figsize=(20,10))

sns.barplot(x=turkey_cities['Province'], y=turkey_cities['Number of Case'])

plt.xticks(rotation= 90)

plt.ylabel('Number of Cases ')

plt.title('Number of Cases in Each Province',fontsize = 50,color='black')

plt.xlabel('date (DAYS)')

plt.grid()
plt.figure(figsize=(20,10))

sns.barplot(x=turkey_cities['Province'].head(15), y=turkey_cities['Number of Case'].head(15))

plt.xticks(rotation= 90)

plt.ylabel('Number of Cases ')

plt.title('Number of Cases in Top 15 Province',fontsize = 50,color='black')

plt.xlabel('date (DAYS)')

plt.grid()
confirmed_tr= turkey['cases'].values.reshape(-1,1)

recovered_tr=turkey['recovered'].values.reshape(-1,1)

deaths_tr=turkey['deaths'].values.reshape(-1,1)

turkey["day"]=range(len(confirmed_tr))

day=turkey["day"].values.reshape(-1,1)
#Polynomial Linear Regression for confirmed cases over time prediction



from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

lr = LinearRegression()



poly = PolynomialFeatures(degree=7)



day_poly = poly.fit_transform(day)



lr.fit(day_poly, confirmed_tr)



predict = lr.predict(day_poly)

plt.figure(figsize=(20,10))

plt.scatter(day, confirmed_tr, color='red')

plt.plot(day, predict, color='blue')

plt.title('Polynomial Linear Regression prediction for Confirmed cases over Time',size=30)

plt.legend(['Actual Confirmed','Predicted Confirmed'])

plt.grid()

plt.show()
#Polynomial Linear Regression for Recovered cases over time prediction prediction



from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

lr = LinearRegression()



poly = PolynomialFeatures(degree=8)



day_poly = poly.fit_transform(day)



lr.fit(day_poly, recovered_tr)



predict = lr.predict(day_poly)

plt.figure(figsize=(20,10))

plt.scatter(day, recovered_tr, color='red')

plt.plot(day, predict, color='blue')

plt.title('Polynomial Linear Regression prediction for Recovered cases over Time',size=30)

plt.legend(['Actual Recovered','Predicted Recovered'])

plt.grid()

plt.show()
#Polynomial Linear Regression prediction for Deaths cases over Time



from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

lr = LinearRegression()



poly = PolynomialFeatures(degree=5)



day_poly = poly.fit_transform(day)



lr.fit(day_poly, deaths_tr)



predict = lr.predict(day_poly)

plt.figure(figsize=(20,10))

plt.scatter(day, deaths_tr, color='red')

plt.plot(day, predict, color='blue')

plt.title('Polynomial Linear Regression prediction for Deaths cases over Time',size=30)

plt.legend(['Actual Deaths','Predicted Deaths'])

plt.grid()

plt.show()


X = turkey[['cases','tests']]

y = turkey['deaths']



from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_train = mms.fit_transform(X_train) 

X_test= mms.fit_transform(X_test)

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=3,p=2,metric='minkowski')



knn = knn.fit(X_train, y_train)



y_pred = knn.predict(X_train)



print("\nPredicted Valus: ",y_pred)

plt.figure(figsize=(20,10))

plt.scatter(y_train, y_pred,c='red')

plt.scatter(y_train, y_train,c='blue')

plt.legend(['Predicted','True'])

plt.xlabel("True Values")

plt.ylabel("Predictions")
# Linear Regression Algorithm



lm = linear_model.LinearRegression()

model = lm.fit(X_train, y_train)

y_true , y_pred =y_test,lm.predict(X_test)



print("\nPredicted Valus: ",y_pred)

plt.figure(figsize=(20,10))

plt.scatter(y_true, y_pred,c='red')

plt.scatter(y_true, y_test,c='blue')

plt.legend(['Predicted','True'])

plt.xlabel("True Values")

plt.ylabel("Predictions")


X = turkey[['cases','tests']]

y = turkey['deaths']



from sklearn.preprocessing import StandardScaler

mms = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_train = mms.fit_transform(X_train) 

X_test= mms.fit_transform(X_test)

#Support Vector machine Algorithm

from sklearn.svm import SVR





SVRModel = SVR(C = 1.0 ,epsilon=0.1,kernel = 'sigmoid')

model = SVRModel.fit(X_train, y_train)

y_true1 , y_pred1 =y_test,SVRModel.predict(X_test)

print("\nPredicted Valus: ",y_pred1)

plt.figure(figsize=(20,10))

plt.scatter(y_true1, y_pred1,c='red')

plt.scatter(y_true1, y_test,c='blue')

plt.legend(['Predicted','True'])

plt.xlabel("True Values")

plt.ylabel("Predictions")
from sklearn.preprocessing import LabelBinarizer



lb = LabelBinarizer()



n_turkey=turkey.copy()

for col in ['cases', 'deaths', 'tests']:

    n_turkey[col] = lb.fit_transform(n_turkey[col])

    

X_data = n_turkey[['tests','cases']]

y_data = n_turkey['deaths']
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=3,p=1,)



knn = knn.fit(X_data, y_data)



y_pred = knn.predict(X_data)



def accuracy(real, predict):

    return sum(y_data == y_pred) / float(real.shape[0])

print(accuracy(y_data, y_pred))
confirmed_cases.head()
recovered_cases.head()
deaths_cases.head()
cols=confirmed_cases.keys()

cols
recovered=recovered_cases.loc[:,cols[4]:cols[-1]]

deaths=deaths_cases.loc[:,cols[4]:cols[-1]]

confirmed=confirmed_cases.loc[:,cols[4]:cols[-1]]

confirmed.head()
dates=confirmed.keys()

world_cases=[]

total_deaths=[]

recoverd_sum=[]

mortality_rate=[]

total_recoverd=[]



for i in dates:

    

    confirmed_sum=confirmed[i].sum()

    death_sum=deaths[i].sum()

    recovered_sum=recovered[i].sum()

    world_cases.append(confirmed_sum)

    total_deaths.append(death_sum)

    mortality_rate.append(death_sum/confirmed_sum)

    total_recoverd.append(recovered_sum)
confirmed_sum,death_sum,recovered_sum
FromDayOne=np.array([i for i in range(len(dates))]).reshape(-1,1)

world_cases=np.array(world_cases).reshape(-1,1)

total_deaths=np.array(total_deaths).reshape(-1,1)

total_recoverd=np.array(total_recoverd).reshape(-1,1)



days_in_fututer=10

future_forecast=np.array([i for i in range(len(dates)+days_in_fututer)]).reshape(-1,1)

ad_dates=future_forecast[:-10]





start='1/22/2020'

start_date=datetime.datetime.strptime(start,"%m/%d/%Y")

future_forecast_dates=[]

for i in range(len(future_forecast)):

     future_forecast_dates.append((start_date+datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

        

latest_confirmed=confirmed_cases[dates[-1]]

latest_deaths=deaths_cases[dates[-1]]

latest_recoverd=recovered_cases[dates[-1]]
unique_countries=list(confirmed_cases['Country/Region'].unique())
country_confirmed_cases=[]

no_cases=[]

for i in unique_countries:

    cases=latest_confirmed[confirmed_cases['Country/Region']==i].sum()

    if cases >0  :

        country_confirmed_cases.append(cases)

    else:

        no_cases.append(cases)



unique_countries=[k for k,v in sorted(zip(unique_countries,country_confirmed_cases), key=operator.itemgetter(1),reverse=True)]

for i in range(len(unique_countries)):

    country_confirmed_cases[i]=latest_confirmed[confirmed_cases['Country/Region']==unique_countries[i]].sum()
for i in range(len(unique_countries)):

    print(f'{unique_countries[i]}  :  {country_confirmed_cases[i]} cases')
unique_provinces=list(confirmed_cases['Province/State'].unique())

province_confirmed_cases=[]

no_cases=[]

for i in unique_provinces:

    cases=latest_confirmed[confirmed_cases['Province/State']==i].sum()

    if cases >0  :

        province_confirmed_cases.append(cases)

    else:

        no_cases.append(cases)

#for i in range(len(no_cases)):

  #unique_provinces.remove(i)
nan_indices=[]

for i in range(len(unique_provinces)):

    if type(unique_provinces[i])==float:

        nan_indices.append(i)

unique_provinces=list(unique_provinces)

province_confirmed_cases=list(province_confirmed_cases)



for i in nan_indices:

    unique_provinces.pop(i)

    province_confirmed_cases.pop(i)
#total number of death overtime

plt.figure(figsize=(20,12))

plt.plot(ad_dates,total_deaths,color='green')

plt.scatter(ad_dates,total_deaths,color='red')

plt.title("Number of Corona Viurse Deaths Over Time (WORLD)",size=30)

plt.xlabel("Time (Days)",size=30)

plt.ylabel('Number of deaths',size=30)

plt.xticks(size=25)

plt.yticks(size=25)

plt.grid()

plt.show()
#total number of Recovered overtime

plt.figure(figsize=(20,12))

plt.plot(ad_dates,total_recoverd,color='blue')

plt.scatter(ad_dates,total_recoverd,color='blue')

plt.title("Number of corona viurse recoverd cases over time (WORLD)",size=30)

plt.xlabel("time",size=30)

plt.ylabel('Number of recoverd',size=30)

plt.xticks(size=15)

plt.yticks(size=15)

plt.grid()

plt.show()
plt.figure(figsize=(20,12))

plt.plot(ad_dates,world_cases,color='blue')

plt.scatter(ad_dates,world_cases,color='blue')

plt.title("Number of Corona Viurse Confirmed Cases Over Time",size=30)

plt.ylabel("Number of total Confirmed",size=30)

plt.xlabel('Date (DAYS)',size=30)

plt.xticks(size=15)

plt.yticks(size=15)

plt.grid()

plt.show()
#total number of death vs recovered overtime

plt.figure(figsize=(20,12))

plt.plot(ad_dates,total_recoverd,color='blue')

plt.scatter(ad_dates,total_recoverd,color='blue')

plt.plot(ad_dates,total_deaths,color='r')

plt.scatter(ad_dates,total_deaths,color='r')

plt.legend(['Recobered','Deaths'],loc='best',fontsize=10)

plt.title("Number of Corona Viurse Recoverd Cases VS Deaths Cases over time (WORLD)",size=30)

plt.xlabel("time",size=30)

plt.ylabel('Number of cases',size=30)

plt.xticks(size=15)

plt.yticks(size=15)

plt.grid()

plt.show()


plt.figure(figsize=(30,15))

plt.subplot(2,2,1)

plt.plot(ad_dates, np.log10(total_deaths))

plt.title('Log of Coronavirus Deaths Over Time (WORLD)',fontsize = 30,color='black')

plt.ylabel('Number of Deaths',size=20)

plt.grid()

plt.subplot(2,2,2)

plt.plot(ad_dates, np.log10(world_cases))

plt.title('Log of Coronavirus Confirmed Cases Over Time (WORLD)',fontsize = 30,color='black')

plt.ylabel('Number of Confirmed',size=20)

plt.grid()

plt.subplot(2,2,3)

plt.plot(ad_dates, np.log10(total_recoverd))

plt.title('Log of Coronavirus Recovered Over Time (WORLD)',fontsize = 30,color='black',)

plt.ylabel('Number of Recovered',size=20)

plt.grid()

#total number of death overtime

mean_mortality_rate=np.mean(mortality_rate)

plt.figure(figsize=(20,12))

plt.plot(ad_dates,mortality_rate,color='blue')

plt.scatter(ad_dates,mortality_rate,color='blue')

plt.axhline(y=mean_mortality_rate,color='red')

plt.legend(['mortality_rate','y='+str(mean_mortality_rate)])

plt.title("Mortality Rate (WORLD)",size=30)

plt.xlabel("time (Days)",size=30)

plt.ylabel('Mortality rate',size=30)

plt.xticks(size=15)

plt.yticks(size=15)

plt.grid()

plt.show()
plt.figure(figsize=(50,20))

sns.barplot(x=unique_countries,y=country_confirmed_cases)

plt.title("Number of Corona Virus Cases in Each Counrty Over Time (World)",size=70)

plt.ylabel(" Number of cases",size=70)

plt.xlabel("Countries Names",size=70)

plt.xticks(rotation= 90)

plt.grid()

plt.show()
visual_unique_conutries=[]

visual_confrimd_cases=[]

others=np.sum(country_confirmed_cases[15:])

for i in range(len(country_confirmed_cases[:15])):

    visual_confrimd_cases.append(country_confirmed_cases[i])

    visual_unique_conutries.append(unique_countries[i])



visual_unique_conutries.append('Others')

visual_confrimd_cases.append(others)
plt.figure(figsize=(20,12))

sns.barplot(x=visual_unique_conutries,y=visual_confrimd_cases,)

plt.title("Top 15 Countries with Confirmed Cases",fontsize=40)
c=random.choices(list(mcolors.CSS4_COLORS.values()),k=len(unique_countries))

plt.figure(figsize=(25,17))

plt.title("Top 15 countries with Confirmed cases")

plt.pie(visual_confrimd_cases,colors=c,labels=visual_unique_conutries,autopct='%1.1f%%',startangle=90)

plt.legend(visual_unique_conutries,loc='upper right')

plt.show()
US_cases = confirmed_cases[confirmed_cases['Country/Region']=='US']

US_cases=US_cases.loc[:,cols[4]:cols[-1]]

US_cases=np.array(US_cases).reshape(-1,1)



US_death = deaths_cases[deaths_cases['Country/Region']=='US']

US_death=US_death.loc[:,cols[4]:cols[-1]]

US_death=np.array(US_death).reshape(-1,1)





US_recovered = recovered_cases[recovered_cases['Country/Region']=='US']

US_recovered=US_recovered.loc[:,cols[4]:cols[-1]]

US_recovered=np.array(US_recovered).reshape(-1,1)
# visual Comparation amoung Recovered , Confirmed , Deaths in US AND TURKEY

plt.figure(figsize=(20,12))

plt.subplot(2,2,1)

plt.plot(ad_dates[55:],US_cases[55:],color='red')

plt.scatter(ad_dates[55:],US_cases[55:],color='red')

plt.title('Confirmed Cases (US)',fontsize = 15,color='black')

plt.ylabel('Number of Cases',size=10)

plt.xlabel('Date (Days)')

plt.grid()

plt.subplot(2,2,2)

plt.plot(ad_dates[55:],US_recovered[55:],color='blue')

plt.scatter(ad_dates[55:],US_recovered[55:],color='blue')

plt.title('Recovered Cases (US)',fontsize = 15,color='black')

plt.ylabel('Number of Recovered',size=10)

plt.xlabel('Date (Days)')

plt.grid()

plt.subplot(2,2,3)

plt.plot(ad_dates[55:],US_death[55:],color='green',)

plt.scatter(ad_dates[55:],US_death[55:],color='green',)

plt.title('Death Cases (US)',fontsize = 15,color='black',)

plt.ylabel('Number of Death',size=10)

plt.xlabel('Date (Days)')

plt.grid()

Egypt_cases = confirmed_cases[confirmed_cases['Country/Region']=='Egypt']

Egypt_cases=Egypt_cases.loc[:,cols[4]:cols[-1]]

Egypt_cases=np.array(Egypt_cases).reshape(-1,1)



Egypt_death = deaths_cases[deaths_cases['Country/Region']=='Egypt']

Egypt_death=Egypt_death.loc[:,cols[4]:cols[-1]]

Egypt_death=np.array(Egypt_death).reshape(-1,1)





Egypt_recovered = recovered_cases[recovered_cases['Country/Region']=='Egypt']

Egypt_recovered=Egypt_recovered.loc[:,cols[4]:cols[-1]]

Egypt_recovered=np.array(Egypt_recovered).reshape(-1,1)
# Deaths,recovered and confirmed cases in egypt

plt.figure(figsize=(20,12))

plt.subplot(2,2,1)

plt.plot(ad_dates[55:],Egypt_cases[55:],color='red')

plt.scatter(ad_dates[55:],Egypt_cases[55:],color='red')



plt.title('Confirmed Cases (Egypt)',fontsize = 15,color='black')

plt.ylabel('Number of Cases',size=10)

plt.xlabel('Date (Days)')

plt.grid()

plt.subplot(2,2,2)

plt.plot(ad_dates[55:],Egypt_recovered[55:],color='blue')

plt.scatter(ad_dates[55:],Egypt_recovered[55:],color='blue')



plt.title('Recovered Cases (Egypt)',fontsize = 15,color='black')

plt.ylabel('Number of Recovered',size=10)

plt.xlabel('Date (Days)')

plt.grid()

plt.subplot(2,2,3)

plt.plot(ad_dates[55:],Egypt_death[55:],color='green')

plt.scatter(ad_dates[55:],Egypt_death[55:],color='green')

plt.title('Death Cases (Egypt)',fontsize = 15,color='black',)

plt.ylabel('Number of Death',size=10)

plt.xlabel('Date (Days)')

plt.grid()


Turkey_cases = confirmed_cases[confirmed_cases['Country/Region']=='Turkey']

Turkey_cases=Turkey_cases.loc[:,cols[4]:cols[-1]]

Turkey_cases=np.array(Turkey_cases).reshape(-1,1)





Turkey_death = deaths_cases[deaths_cases['Country/Region']=='Turkey']

Turkey_death=Turkey_death.loc[:,cols[4]:cols[-1]]

Turkey_death=np.array(Turkey_death).reshape(-1,1)





Turkey_recovered = recovered_cases[recovered_cases['Country/Region']=='Turkey']

Turkey_recovered=Turkey_recovered.loc[:,cols[4]:cols[-1]]

Turkey_recovered=np.array(Turkey_recovered).reshape(-1,1)
# Deaths,recovered and confirmed cases in turkey

plt.figure(figsize=(20,12))

plt.subplot(2,2,1)

plt.plot(ad_dates[55:],Turkey_cases[55:],color='red')

plt.scatter(ad_dates[55:],Turkey_cases[55:],color='red')



plt.title('Confirmed Cases (Turkey)',fontsize = 15,color='black')

plt.ylabel('Number of Cases',size=10)

plt.xlabel('Date (Days)')

plt.grid()

plt.subplot(2,2,2)

plt.plot(ad_dates[55:],Turkey_recovered[55:],color='blue')

plt.scatter(ad_dates[55:],Turkey_recovered[55:],color='blue')



plt.title('Recovered Cases (Turkey)',fontsize = 15,color='black')

plt.ylabel('Number of Recovered',size=10)

plt.xlabel('Date (Days)')

plt.grid()

plt.subplot(2,2,3)

plt.plot(ad_dates[55:],Turkey_death[55:],color='green')

plt.scatter(ad_dates[55:],Turkey_death[55:],color='green')

plt.title('Death Cases (Turkey)',fontsize = 15,color='black',)

plt.ylabel('Number of Death',size=10)

plt.xlabel('Date (Days)')

plt.grid()
# visual Comparation amoung Recovered , Confirmed , Deaths in US,TURKEY AND EGYPT

plt.figure(figsize=(20,10))

plt.subplot(2,2,1)

plt.plot(ad_dates[55:],US_cases[55:],color='red')

plt.scatter(ad_dates[55:],US_cases[55:],color='red')

plt.plot(ad_dates[55:],Turkey_cases[55:],color='green')

plt.scatter(ad_dates[55:],Turkey_cases[55:],color='green')

plt.plot(ad_dates[55:],Egypt_cases[55:],color='blue')

plt.scatter(ad_dates[55:],Egypt_cases[55:],color='blue')



plt.title('Confirmed Cases (US VS TURKEY VS EGYPT)',fontsize = 15,color='black')

plt.ylabel('Number of Deaths',size=10)

US = mpatches.Patch(color='red', label='US Confirmed CASES')

TURKEY = mpatches.Patch(color='green', label='TURKEY Confirmed CASES')

EGYPT = mpatches.Patch(color='blue', label='EGYPT Confirmed CASES')

plt.legend(handles=[US,TURKEY,EGYPT])

plt.subplot(2,2,2)

plt.plot(ad_dates[55:],US_recovered[55:],color='red')

plt.scatter(ad_dates[55:],US_recovered[55:],color='red')

plt.plot(ad_dates[55:],Turkey_recovered[55:],color='green')

plt.scatter(ad_dates[55:],Turkey_recovered[55:],color='green')

plt.plot(ad_dates[55:],Egypt_recovered[55:],color='blue')

plt.scatter(ad_dates[55:],Egypt_recovered[55:],color='blue')

plt.title('Recovered Cases (US VS TURKEY VS EGYPT)',fontsize = 15,color='black')

plt.ylabel('Number of Confirmed',size=10)

US = mpatches.Patch(color='red', label='US Recovered CASES')

TURKEY = mpatches.Patch(color='green', label='TURKEY Recovered CASES')

EGYPT = mpatches.Patch(color='blue', label='EGYPT Recovered CASES')

plt.legend(handles=[US,TURKEY,EGYPT])

plt.subplot(2,2,3)

plt.plot(ad_dates[55:],US_death[55:],color='red')

plt.scatter(ad_dates[55:],US_death[55:],color='red')

plt.plot(ad_dates[55:],Turkey_death[55:],color='green')

plt.scatter(ad_dates[55:],Turkey_death[55:],color='green')



plt.plot(ad_dates[55:],Egypt_death[55:],color='blue')

plt.scatter(ad_dates[55:],Egypt_death[55:],color='blue')



plt.title('Death Cases (US VS TURKEY VS EGYPT)',fontsize = 15,color='black',)

plt.ylabel('Number of Recovered',size=10)

US = mpatches.Patch(color='red', label='US Death CASES')

TURKEY = mpatches.Patch(color='green', label='TURKEY Death CASES')

EGYPT = mpatches.Patch(color='blue', label='EGYPT Death CASES')

plt.legend(handles=[US,TURKEY,EGYPT])
Turkey_confrimed=latest_confirmed[confirmed_cases['Country/Region']=='Turkey'].sum()

out_Turkey=np.sum(country_confirmed_cases)-Turkey_confrimed

plt.figure(figsize=(16,9))

plt.barh("Turkey Confrimed",Turkey_confrimed)

plt.barh("Outside Turkey confrimed",out_Turkey)

plt.title("Turkey vs Rest of the World")
print('Outside Turkey {} cases '.format(out_Turkey))

print('Turkey_confrimed {} cases '.format(Turkey_confrimed))

print('total {} cases '.format(Turkey_confrimed+out_Turkey))
US_confrimed=latest_confirmed[confirmed_cases['Country/Region']=='US'].sum()

out_US=np.sum(country_confirmed_cases)-US_confrimed

plt.figure(figsize=(16,9))

plt.barh("US Confrimed",US_confrimed)

plt.barh("Outside US confrimed",out_US)

plt.title("US vs Rest of the World")
print('Outside US {} cases '.format(out_US))

print('US_confrimed {} cases '.format(US_confrimed))

print('total {} cases '.format(US_confrimed+out_US))
Egypt_confrimed=latest_confirmed[confirmed_cases['Country/Region']=='Egypt'].sum()

out_Egypt=np.sum(country_confirmed_cases)-Egypt_confrimed

plt.figure(figsize=(16,9))

plt.barh("Egypt Confrimed",Egypt_confrimed)

plt.barh("Outside Egypt confrimed",out_Egypt)

plt.title("Egypt vs Rest of the World")
print('Outside Egypt {} cases '.format(out_Egypt))

print('Egypt_confrimed {} cases '.format(Egypt_confrimed))

print('total {} cases '.format(Egypt_confrimed+out_Egypt))
#split data for future perdiction (World)



X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(FromDayOne, world_cases, test_size=0.15, shuffle=True)
kernel=['poly','sigmoid','rbf']

c=[0.01,0.1,1,10]

gamma=[0.01,0.1,1]

epsilon=[0.01,0.1,1]

shrinking=[True,False]

world_svm_grid={'kernel':kernel,'C':c,'gamma':gamma,'epsilon':epsilon,'shrinking':shrinking}



world_svm=SVR()

world_svm_search=RandomizedSearchCV(world_svm,world_svm_grid,scoring='neg_mean_squared_error',cv=3,return_train_score=True,n_jobs=-1,n_iter=10,verbose=1)



world_svm_search.fit(X_train_confirmed,y_train_confirmed.ravel())

print(world_svm_search.best_params_)

world_svm_confirmed=world_svm_search.best_estimator_

world_svm_pred=world_svm_confirmed.predict(future_forecast)

print(world_svm_confirmed)

world_svm_pred





from sklearn.metrics import median_absolute_error

world_svm_test_pred=world_svm_confirmed.predict(X_test_confirmed)

print('Mean Absolute Error :',mean_absolute_error(world_svm_test_pred,y_test_confirmed))

print('Mean Squared Error :',mean_squared_error(world_svm_test_pred,y_test_confirmed))

print('Media Absolute Error :',median_absolute_error(world_svm_test_pred,y_test_confirmed))

plt.figure(figsize=(22,14))

plt.subplot(2,2,1)

plt.plot(ad_dates,world_cases,marker='o', linestyle='dashed')



plt.title("Number of corona viurse cases over time (Actual)",size=15)

plt.xlabel("days",size=15)

plt.ylabel("Number of cases",size=15)

plt.xticks(size=15)

plt.yticks(size=14)



plt.subplot(2,2,2)

plt.plot(future_forecast,world_svm_pred,marker='o', linestyle='dashed',color='red')



plt.title("Number of corona viurse cases over time (predicted)",size=15)

plt.xlabel("days",size=15)

plt.ylabel("Number of cases",size=15)

plt.xticks(size=15)

plt.yticks(size=14)



plt.subplot(2,2,3)

plt.title("world svm test pred vs y test confirmed",size=15)

plt.plot(world_svm_test_pred)

plt.plot(y_test_confirmed)

plt.legend(["world svm test pred",'y test confirmed'])

plt.subplot(2,2,4)

plt.plot(ad_dates,world_cases,color='blue')

plt.plot(future_forecast,world_svm_pred,linestyle='dashed',color='red')



plt.title("World Actual cases ve Predicted (SVM MODEL)",size=15)

plt.xlabel("days",size=15)

plt.ylabel("Number of cases",size=15)

plt.xticks(size=25)

plt.yticks(size=25)

plt.legend(["World Confirmed cases",'World SVM Predctions'])
#predict cases for the next 10 dayes



print('SVM future prediction For Next 10 Days Over The World')

set(zip(future_forecast_dates[-10:],world_svm_pred[-10:]))
from sklearn.linear_model import LinearRegression

world_linear_model=LinearRegression(normalize=True,fit_intercept=True)

world_linear_model.fit(X_train_confirmed,y_train_confirmed)

lr_world_test_pred=world_linear_model.predict(X_test_confirmed)

world_linear_pred=world_linear_model.predict(future_forecast)

print('Mean Absolute Error :',mean_absolute_error(lr_world_test_pred,y_test_confirmed))

print('Mean Squared Error :',mean_squared_error(lr_world_test_pred,y_test_confirmed))

print('Media Absolute Error :',median_absolute_error(lr_world_test_pred,y_test_confirmed))
plt.figure(figsize=(22,14))

plt.subplot(2,2,1)

plt.plot(ad_dates,world_cases,marker='o', linestyle='dashed')



plt.title("Number of corona viurse cases over time (Actual)",size=15)

plt.xlabel("days",size=15)

plt.ylabel("Number of cases",size=15)

plt.xticks(size=15)

plt.yticks(size=14)



plt.subplot(2,2,2)

plt.plot(future_forecast,world_linear_pred,marker='o', linestyle='dashed',color='red')



plt.title("Number of corona viurse cases over time (predicted)",size=15)

plt.xlabel("days",size=15)

plt.ylabel("Number of cases",size=15)

plt.xticks(size=15)

plt.yticks(size=14)



plt.subplot(2,2,3)

plt.title("World Linear Regression test pred vs y Test Confirmed",size=15)

plt.plot(lr_world_test_pred)

plt.plot(y_test_confirmed)

plt.legend(["World Linear regression test pred",'y test confirmed'])

plt.subplot(2,2,4)

plt.plot(ad_dates,world_cases,color='blue')

plt.plot(future_forecast,world_linear_pred,linestyle='dashed',color='red')



plt.title("World Actual cases vs Predicted cases (Linear Regression MODEL)",size=15)

plt.xlabel("days",size=15)

plt.ylabel("Number of cases",size=15)

plt.xticks(size=25)

plt.yticks(size=25)

plt.legend(["World Confirmed cases",'World Linear Regression Predctions'])
#predict cases for the next 10 dayes



print('Linear Regression Model future prediction (WOLRD CASES)')

print(' Number Of Cases  :\n\n',world_linear_pred[-10:])
#split data for future perdiction (Turkey)



X_train_turkey, X_test_turkey, y_train_turkey, y_test_turkey = train_test_split(FromDayOne[55:], Turkey_cases[55:], test_size=0.15, shuffle=True)
kernel=['poly','sigmoid','rbf']

c=[0.01,0.1,1,10]

gamma=[0.01,0.1,1]

epsilon=[0.01,0.1,1]

shrinking=[True,False]

turkey_svm_grid={'kernel':kernel,'C':c,'gamma':gamma,'epsilon':epsilon,'shrinking':shrinking}



turkey_svm=SVR()

turkey_svm_search=RandomizedSearchCV(turkey_svm,turkey_svm_grid,scoring='neg_mean_squared_error',cv=3,return_train_score=True,n_jobs=-1,n_iter=7,verbose=1)



print(turkey_svm_search.fit(X_train_turkey,y_train_turkey.ravel()))

turkey_svm_confirmed=turkey_svm_search.best_estimator_

turkey_svm_pred=turkey_svm_confirmed.predict(future_forecast)

print(turkey_svm_confirmed)

turkey_svm_pred
turkey_svm_test_pred=turkey_svm_confirmed.predict(X_test_turkey)

print('Mean Absolute Error :',mean_absolute_error(turkey_svm_test_pred,y_test_turkey))

print('Mean Squared Error :',mean_squared_error(turkey_svm_test_pred,y_test_turkey))

print('Media Absolute Error :',median_absolute_error(turkey_svm_test_pred,y_test_turkey))
plt.figure(figsize=(22,14))

plt.subplot(2,2,1)

plt.plot(ad_dates[55:],Turkey_cases[55:],marker='o', linestyle='dashed')



plt.title("Number of corona viurse cases over time (Actual)",size=15)

plt.xlabel("days",size=15)

plt.ylabel("Number of cases",size=15)

plt.xticks(size=15)

plt.yticks(size=14)



plt.subplot(2,2,2)

plt.plot(future_forecast,turkey_svm_pred,marker='o', linestyle='dashed',color='red')



plt.title("Number of corona viurse cases over time (predicted)",size=15)

plt.xlabel("days",size=15)

plt.ylabel("Number of cases",size=15)

plt.xticks(size=15)

plt.yticks(size=14)



plt.subplot(2,2,3)

plt.title("Turkey svm test pred vs y test confirmed",size=15)

plt.plot(turkey_svm_test_pred)

plt.plot(y_test_turkey)

plt.legend(["Turkey svm test pred",'y test confirmed'])

plt.subplot(2,2,4)

plt.plot(ad_dates[55:],Turkey_cases[55:],color='blue')

plt.plot(future_forecast,turkey_svm_pred,linestyle='dashed',color='red')



plt.title("Turkey Actual cases vs Predicted cases (SVM MODEL)",size=15)

plt.xlabel("days",size=15)

plt.ylabel("Number of cases",size=15)

plt.xticks(size=15)

plt.yticks(size=15)

plt.legend(["Turkey Confirmed cases",'TURKEY SVM Predctions'])
print('SVM future prediction For Next 10 Days in Turkey')

print('Day  :  Cases')

set(zip(future_forecast_dates[-10:],turkey_svm_pred[-10:]))
from sklearn.linear_model import LinearRegression

turkey_linear_model=LinearRegression(normalize=True,fit_intercept=True,n_jobs=-1,)

turkey_linear_model.fit(X_train_turkey,y_train_turkey)

lr_turkey_test_pred=turkey_linear_model.predict(X_test_turkey)

turkey_linear_pred=turkey_linear_model.predict(future_forecast)

print('Mean Absolute Error :',mean_absolute_error(lr_turkey_test_pred,y_test_turkey))

print('Mean Squared Error :',mean_squared_error(lr_turkey_test_pred,y_test_turkey))

print('Media Absolute Error :',median_absolute_error(lr_turkey_test_pred,y_test_turkey))
plt.figure(figsize=(22,14))

plt.subplot(2,2,1)

plt.plot(ad_dates[55:],Turkey_cases[55:],marker='o', linestyle='dashed')



plt.title("Number of corona viurse cases over time (Actual)",size=15)

plt.xlabel("days",size=15)

plt.ylabel("Number of cases",size=15)

plt.xticks(size=15)

plt.yticks(size=14)



plt.subplot(2,2,2)

plt.plot(future_forecast,turkey_linear_pred,marker='o', linestyle='dashed',color='red')



plt.title("Number of corona viurse over cases time (predicted)",size=15)

plt.xlabel("days",size=15)

plt.ylabel("Number of cases",size=15)

plt.xticks(size=15)

plt.yticks(size=14)



plt.subplot(2,2,3)

plt.title("Turkey Linear Regression test pred vs y Test Confirmed",size=15)

plt.plot(lr_turkey_test_pred)

plt.plot(y_test_turkey)

plt.legend(["Turkey Linear regression test pred",'y test confirmed'])

plt.subplot(2,2,4)

plt.plot(ad_dates[55:],Turkey_cases[55:],color='blue')

plt.plot(future_forecast,turkey_linear_pred,linestyle='dashed',color='red')



plt.title("Turkey Actual cases vs Predicted cases (Linear Regression MODEL)",size=15)

plt.xlabel("days",size=15)

plt.ylabel("Number of cases",size=15)

plt.xticks(size=25)

plt.yticks(size=25)

plt.legend(["Turkey Confirmed cases",'Turkey Linear Regression Predctions'])
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

lr = LinearRegression()



poly = PolynomialFeatures(degree=8)



day_poly = poly.fit_transform(ad_dates)



lr.fit(day_poly, Turkey_cases)



predict = lr.predict(day_poly)

plt.figure(figsize=(20,10))

plt.scatter(ad_dates, Turkey_cases, color='red')

plt.plot(ad_dates, predict, color='blue')

plt.legend(['Confirmed','Predicted'])

plt.show()