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
import datetime

df_patient=pd.read_csv('../input/coronavirusdataset/patient.csv')

dt_now = datetime.datetime.now()

this_year=dt_now.year



df_patient=df_patient[df_patient['country']=='Korea']

df_patient['age']=2020-df_patient['birth_year']

df_patient['confirmed_date'] = pd.to_datetime(df_patient['confirmed_date'],format="%Y-%m-%d")

df_patient['released_date'] = pd.to_datetime(df_patient['released_date'],format="%Y-%m-%d")

df_patient['deceased_date'] = pd.to_datetime(df_patient['deceased_date'],format="%Y-%m-%d")



df_patient=df_patient.drop(columns=['group','infected_by','birth_year','infection_order','disease','country'])



df_patient.head()
age_range=[]

for i in df_patient['age']:

    if i>=0 and i<=10:

        age_range.append('0~10')

    elif i>10 and i<=20:

        age_range.append('11~20')

    elif i>20 and i<=30:

        age_range.append('21~30')

    elif i>30 and i<=40:

        age_range.append('31~40')

    elif i>40 and i<=50:

        age_range.append('41~50')

    elif i>50 and i<=60:

        age_range.append('51~60')

    elif i>60 and i<=70:

        age_range.append('61~70')

    elif i>70 and i<=80:

        age_range.append('71~80')

    elif i>80 and i<=90:

        age_range.append('81~90')

    elif i>90 and i<=100:

        age_range.append('91~100')

    elif i>100 and i<=110:

        age_range.append('101~110')



df_patient['age']=pd.DataFrame(age_range)

df_patient['Incubation_period(released)']=df_patient['released_date']-df_patient['confirmed_date']

df_patient['Incubation_period(deceased)']=df_patient['deceased_date']-df_patient['confirmed_date']

df_patient.head()





#df_patient1 = df_patient.groupby('age')[''].sum()
df_patient1=df_patient.loc[:,['age','Incubation_period(released)','Incubation_period(deceased)','state']]

df_patient1=pd.get_dummies(df_patient1,columns=['state'])#drop_first=True

df_patient1.head()
age=pd.DataFrame(df_patient1['age'].value_counts()).reset_index()

age = age.rename(columns={'age':'number_patients','index':'age'})

age=age.sort_values('age', ascending=True).reset_index()

age=age.drop(columns=['index','age'])

df_patient1= df_patient1.groupby('age')['state_deceased','state_isolated','state_released'].sum().reset_index()

df_patient1=pd.concat([df_patient1,age],axis=1,sort=True)



df_patient1_1_1=df_patient1.style.background_gradient(cmap='Reds')

df_patient1_1_1



#age
df_patient1_1=df_patient1.drop(columns=['age'])

import seaborn as sns



sns.set(style='darkgrid')



import matplotlib.pyplot as plt

plt.figure(figsize=(30, 8))

bar_width = 0.1



x = np.arange(len(df_patient1_1), dtype='float64')



for col in df_patient1_1.columns:

    plt.bar(x, df_patient1_1[col], width=bar_width, label=col)

    x+= bar_width

    

plt.xticks((np.arange(len(df_patient1_1)) + x - bar_width) / 2,df_patient1['age'], fontsize=20)



plt.xlabel('age',fontsize=20)

plt.ylabel('number',fontsize=20)

plt.legend(fontsize=20)

plt.show()
df_patient2= df_patient.groupby('age')['Incubation_period(released)'].sum().reset_index()

df_patient2
df_patient2['Incubation_period(released)']=df_patient2['Incubation_period(released)']/df_patient1['state_released']

df_patient2['Incubation_period(released)']=df_patient2['Incubation_period(released)'].astype('int')

df_patient2=df_patient2[df_patient2['Incubation_period(released)']>0]

df_patient2['Incubation_period(released)']=round(df_patient2['Incubation_period(released)']/100000000000000,1)

df_patient2_1=df_patient2.style.background_gradient(cmap='Reds')

df_patient2_1
sns.set(style='darkgrid')

fig = plt.figure(figsize=(12,8))



plt.xticks(rotation=60)

ax = sns.lineplot(df_patient2['age'],df_patient2['Incubation_period(released)'],color='red')



plt.xlabel('age',fontsize=15)

plt.ylabel('Incubation_period(day)',fontsize=15)

ax.set_yticks(np.linspace(0, 20, 21))



from matplotlib import ticker

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
df_patient_3=df_patient[df_patient['sex']=='male']

df_patient_4=df_patient[df_patient['sex']=='female']



#male_ave_age=pd.DataFrame(df_patient_3['age'].value_counts()).reset_index()

#male_ave_age=male_ave_age.rename(columns={'age':'count','index':'age'})

#male_ave_age=male_ave_age.sort_values('age').reset_index(drop=True)



#female_ave_age=pd.DataFrame(df_patient_4['age'].value_counts()).reset_index()

#female_ave_age=female_ave_age.rename(columns={'age':'count','index':'age'})

#female_ave_age=female_ave_age.sort_values('age').reset_index(drop=True)
#sns.set(style='darkgrid')

#fig = plt.figure(figsize=(12,8))



#plt.xticks(rotation=60)

#ax = sns.lineplot(female_ave_age['age'],female_ave_age['count'],color='red')

#ax = sns.lineplot(male_ave_age['age'],male_ave_age['count'],color='blue')



#plt.xlabel('age',fontsize=15)

#plt.ylabel('number_patients',fontsize=15)



#ax.set_ticks(np.linspace(0, 25, 26))

#plt.legend(['female','male'],loc='best',fontsize=16)

#from matplotlib import ticker

#ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
Date=df_patient_3['confirmed_date'].value_counts()

Date=pd.DataFrame(Date)

Date=Date.reset_index()

Date=pd.DataFrame(Date['index'])

Date=Date.rename(columns={'index':'date'})

Date=Date.sort_values('date').reset_index(drop=True)

Date.head()
male_accum=[]

accum=0

for col in Date['date']:

    count=df_patient_3[df_patient_3['confirmed_date']==col]

    accum=accum+len(count)

    male_accum.append(accum)

    

male_accum=pd.DataFrame(male_accum,columns=['accum_patients'])

male_accum=pd.concat([Date,male_accum], axis=1, sort=True)

male_accum
female_accum=[]

accum1=0

for col in Date['date']:

    count1=df_patient_4[df_patient_4['confirmed_date']==col]

    accum1=accum1+len(count1)

    female_accum.append(accum1)

    

female_accum=pd.DataFrame(female_accum,columns=['accum_patients'])

female_accum=pd.concat([Date,female_accum], axis=1, sort=True)

female_accum
sns.set(style='darkgrid')

fig = plt.figure(figsize=(12,8))



plt.xticks(rotation=60)

ax = sns.lineplot(male_accum['date'],male_accum['accum_patients'],color='blue')

ax = sns.lineplot(female_accum['date'],female_accum['accum_patients'],color='red')

plt.xlabel('date',fontsize=15)

plt.ylabel('accum_patients',fontsize=15)



#ax.set_yticks(np.linspace(0, 25, 26))

plt.legend(['male','female'],loc='best',fontsize=16)

from matplotlib import ticker

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
from fbprophet import Prophet

from plotly.offline import init_notebook_mode, iplot, plot

from matplotlib import ticker

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))



male_accum=male_accum.rename(columns={'date':'ds','accum_patients':'y'})

female_accum=female_accum.rename(columns={'date':'ds','accum_patients':'y'})



male=Prophet()

female=Prophet()



male.fit(male_accum)

female.fit(female_accum)



future_male=male.make_future_dataframe(periods=30)

future_female=female.make_future_dataframe(periods=30)



forecast_male=male.predict(future_male)

forecast_female=female.predict(future_female)



figure_male=male.plot(forecast_male,xlabel='Date',ylabel='Confirmed Count(male)')

figure_female=female.plot(forecast_female,xlabel='Date',ylabel='Confirmed Count(female)')
male_death=df_patient_3[df_patient_3['state']=='deceased']

female_death=df_patient_4[df_patient_4['state']=='deceased']
count_male_death=len(male_death)

count_female_death=len(female_death)



rate_male_death=count_male_death/accum

rate_female_death=count_female_death/accum1



death_rate=pd.DataFrame(data=[[rate_male_death,rate_female_death]],columns=['male','female'])
plt.figure(figsize=(12,8))

plt.bar('male', death_rate['male'], width=0.7, label='male')

plt.bar('female', death_rate['female'], width=0.7, label='female')

plt.xlabel('sex',fontsize=15)

plt.ylabel('death_rate(×100%)',fontsize=15)

fig.show()
df_patient.head()
reason=df_patient['infection_reason']

reason=reason.dropna()

reason=df_patient['infection_reason'].value_counts()

reason=pd.DataFrame(reason).rename(columns={"infection_reason":"number"})

reason=reason.sort_values("number", ascending=True)

reason
plt.figure(figsize=(12,8))

plt.barh(reason.index,reason['number'],label='infection_reason',color='red')

plt.xlabel('number_patients',fontsize=15)

plt.show()
contact=pd.DataFrame(df_patient['contact_number'])

contact=contact.dropna()

infection_rate=len(contact)/contact['contact_number'].sum()

print('Infection probability by contact with patients:',format(round(infection_rate*100,2)),'%')
df_route=pd.read_csv('../input/coronavirusdataset/route.csv')

place=pd.DataFrame(df_route['visit'])

place=pd.DataFrame(place['visit'].value_counts())

place=place.sort_values("visit", ascending=True)





plt.figure(figsize=(12,8))

plt.barh(place.index,place['visit'],label='place.index',color='green')

plt.xlabel('number_patients',fontsize=15)

plt.show()
df_time=pd.read_csv('../input/coronavirusdataset/time.csv')

df_time=df_time.iloc[:,:7]

df_time.head()
fig = plt.figure(figsize=(34,20))

#ax = sns.line

now_infection_number=pd.DataFrame(df_time['confirmed']-df_time['released']-df_time['deceased'],columns=['now_infection_number'])



plt.subplot(121)

plt.xticks(rotation=60)

plt.plot(df_time['date'],df_time['test'],color='black')

plt.plot(df_time['date'],df_time['negative'],color='blue')

plt.plot(df_time['date'],df_time['confirmed'],color='red')



plt.xlabel('date',fontsize=15)

plt.ylabel('number',fontsize=15)

plt.legend(['test','negative','comfirmed'],loc='best',fontsize=16)



plt.subplot(122)

plt.xticks(rotation=60)

plt.plot(df_time['date'],df_time['released'],color='blue')

plt.plot(df_time['date'],df_time['deceased'],color='red')

plt.plot(df_time['date'],now_infection_number['now_infection_number'],color='green')

plt.xlabel('date',fontsize=15)

plt.ylabel('number',fontsize=15)

plt.legend(['released','deceased','during treatment'],loc='best',fontsize=16)



plt.show()
df_time=pd.concat([df_time,now_infection_number],axis=1,sort=True)

df_time=df_time.rename(columns={"date":"ds"})

df_time.head()
during_treatment=df_time.loc[:,['ds','confirmed']].rename(columns={"confirmed":"y"})

released=df_time.loc[:,['ds','released']].rename(columns={"released":"y"})

deceased=df_time.loc[:,['ds','deceased']].rename(columns={"deceased":"y"})
from fbprophet import Prophet





m=Prophet()

m.fit(during_treatment.iloc[30:,])

future=m.make_future_dataframe(periods=10)

forecast=m.predict(future)



m1=Prophet()

m1.fit(released.iloc[30:,])

future1=m1.make_future_dataframe(periods=10)

forecast1=m1.predict(future1)



m2=Prophet()

m2.fit(deceased.iloc[30:,])

future2=m2.make_future_dataframe(periods=10)

forecast2=m2.predict(future2)



figure = m.plot(forecast,xlabel='Date',ylabel='during_treatment')



figure1 = m1.plot(forecast1,xlabel='Date',ylabel='released')



figure2 = m2.plot(forecast2,xlabel='Date',ylabel='deceased')

df_trend=pd.read_csv('../input/coronavirusdataset/trend.csv')

df_trend
plt.figure(figsize=(24,8))

plt.xticks(rotation=60)

plt.plot(df_trend['date'],df_trend['cold'],color='blue')

plt.plot(df_trend['date'],df_trend['flu'],color='yellow')

plt.plot(df_trend['date'],df_trend['pneumonia'],color='green')

plt.plot(df_trend['date'],df_trend['coronavirus'],color='red')

plt.xlabel('date',fontsize=15)

plt.ylabel('trend',fontsize=15)

plt.legend(['cold','flu','pneumonia','coronavirus'],loc='best',fontsize=16)
corona=df_trend.loc[:,['date','coronavirus']].rename(columns={"date":"ds","coronavirus":"y"})
from fbprophet import Prophet

m=Prophet()

m.fit(corona.loc[42:,])

future=m.make_future_dataframe(periods=10)

forecast=m.predict(future)



figure = m.plot(forecast,xlabel='Date',ylabel='trend')