

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import linear_model

import xlrd 

import numpy as np

import seaborn as sns

%matplotlib inline
filename="../input/ghana-covid19-dataset/Ghana_Covid19_DailyActive.csv"

data=pd.read_csv(filename)

data.head()
data.tail()
print(f'Date range: {len(data.date.unique())} days')

print('FROM', min(data.date), 'to', max(data.date))
data.shape
data.info()
data.isnull().sum()
data=data.rename(columns={'confirmed':'confirmedCases','recovered':'recoveredCases','death':'death','cumulative_confirmed ':'cumulativeConfirmed ','chol':'Chol',

                   'cumulative_recovered':'cumulativeRecovered','cumulative_death':'cumulativeDeath','active_cases':'activeCases'})
data.columns
data.describe()


print(f'The total number of CONFIRMED Cases for the past 139 days is :{(data.confirmedCases.sum())}')

print(f'The total number of Recovery Cases for the past 139 days is :{(data.recoveredCases.sum())}')

print(f'The total number of DEATH Cases for the past 139 days is :{(data.death.sum())}')

print(f'The current ACTIVE Cases for the past 139 days is :{(data.confirmedCases.sum()-data.recoveredCases.sum())-data.death.sum()}')







confirmed=data.confirmedCases.sum()

recovery=data.recoveredCases.sum()

active=(data.confirmedCases.sum()-data.recoveredCases.sum())-data.death.sum()

death=data.death.sum()
sns.barplot(x=["Confirmed Cases","Recovery Cases","Active Cases","Death Cases"],y=[confirmed,recovery,active,death])

plt.title("TOTAL CASES")

plt.xlabel("VARIOUS CASES")

plt.ylabel("NUMBER")

plt.show()
plt.style.use('ggplot')

plt.style.use('dark_background')



fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(23, 7))

ax1, ax2, ax3, ax4 = axes.ravel()



ax1.set_title('Confirmed Cases', fontsize=15)

ax1.plot(data.confirmedCases, color='b')



ax2.set_title('Recovered Cases', fontsize=15)

ax2.plot(data.recoveredCases, color='g')





ax3.set_title('Active Cases', fontsize=15)

ax3.plot(data.activeCases, color='y')



ax4.set_title('Death Cases', fontsize=15)

ax4.plot(data.death, color='r')







plt.show()


fig, ax = plt.subplots(figsize=(13, 7))

plt.title('Active & Recovered  cases graph ', fontsize=15)

ax.set_xlabel('Date', size=13)

ax.set_ylabel('Number of cases', size=13)

plt.plot(data.activeCases,label='Active Cases',color='r')

plt.plot(data.date.unique(),data.recoveredCases,label='Recovered',color='y')

ax.set_xticks(ax.get_xticks()[::7])

plt.xticks(rotation=70)

ax.legend()

plt.show()
new_data=pd.read_csv(filename, parse_dates=['date'],index_col='date')

new_data.head(5)
new_data.tail(5)
new_data.info()
new_data.index
march_only=new_data["3/2020"]

april_only=new_data["4/2020"]

may_only=new_data["5/2020"]

june_only=new_data["6/2020"]

july_only=new_data["7/2020"]

august_only=new_data["8/2020"]







fig, ax = plt.subplots(figsize=(13, 7))

plt.title('General cases ', fontsize=15)

plt.style.use('dark_background')

ax.set_xlabel('Date', size=13)

ax.set_ylabel('Number of cases', size=13)

plt.step(march_only.index,march_only['active_cases'], label='MARCH ACTIVE')

plt.step(april_only.index,april_only['active_cases'],label='APRIL ACTIVE')

plt.step(may_only.index,may_only['active_cases'],label='MAY ACTIVE')

plt.step(june_only.index,june_only['active_cases'],label='JUNE ACTIVE')

plt.step(july_only.index,july_only['active_cases'],label='JULY ACTIVE')

plt.plot(august_only['active_cases'],label='AUGUST ACTIVE')



plt.step(march_only.index,march_only['recovered'],label='MARCG RECOVERED')

plt.step(april_only.index,april_only['recovered'],label='APRIL RECOVERED')

plt.step(may_only.index,may_only['recovered'],label='MAY RECOVERED')

plt.step(june_only.index,june_only['recovered'],label='JUNE RECOVERED')

plt.step(july_only.index,july_only['recovered'],label='JULY RECOVERED')

plt.plot(august_only['recovered'],label='AUGUST RECOVERED')





ax.set_xticks(ax.get_xticks()[::7])

ax.legend()

plt.show()
fig, ax = plt.subplots(figsize=(13, 7))

plt.title('CUMULATIVE DEATH ', fontsize=15)

ax.set_xlabel('Date', size=13)

ax.set_ylabel('Number of cases', size=13)



plt.plot(march_only.index,march_only['cumulative_death'],'ro',label='MARCH')

plt.plot(april_only.index,april_only['cumulative_death'],'bo',label='APRIL')

plt.plot(may_only.index,may_only['cumulative_death'],'go',label='MAY')

plt.plot(june_only.index,june_only['cumulative_death'],'yo',label='JUNE')

plt.plot(july_only.index,july_only['cumulative_death'],'r*',label='JULY')

plt.plot(august_only.index,august_only['cumulative_death'],'b^',label='JULY')



ax.set_xticks(ax.get_xticks()[::7])

ax.legend()

plt.show()
march_only.head()
print(f'The total number of Death Cases in MARCH is :{(march_only.death.sum())}')

print(f'The total number of Death Cases in APRIL is :{(april_only.death.sum())}')

print(f'The total number of Death Cases in MAY is :{(may_only.death.sum())}')

print(f'The total number of Death Cases in JUNE is :{(june_only.death.sum())}')

print(f'The total number of Death Cases in JUNE is :{(july_only.death.sum())}')

print(f'The total number of Death Cases in JUNE is :{(august_only.death.sum())}')
plt.figure(figsize=(20,10))

plt.title('MONTHLY DEATH CASES', fontsize=15)

sns.barplot(x=["MARCH","APRIL","MAY","JUNE","JULY","AUGUST"],y=[march_only.death.sum(),april_only.death.sum(),may_only.death.sum(),june_only.death.sum(),july_only.death.sum(),august_only.death.sum()])

plt.xlabel("VARIOUS MONTHS")

plt.ylabel("TOTAL DEATH")

plt.show()
Lockdown_period=new_data["30/3/2020":"20/4/2020"]

Lockdown_period.head()

Lconfrimed_cases=Lockdown_period["confirmed"].sum()

Lactive_cases=(Lockdown_period["confirmed"].sum()-Lockdown_period["recovered"].sum())-Lockdown_period["death"].sum()

Lrecovery_cases=Lockdown_period["recovered"].sum()

Ldeath_cases=Lockdown_period["death"].sum()
plt.figure(figsize=(20,10))

plt.title("LOCKDOWN PERIOD")

plt.xlabel("CASES")

plt.ylabel("Number_Frequency")

sns.barplot(x=["CONFIRMED","ACTIVE","RECOVERY","DEATH"],y=[Lconfrimed_cases,Lactive_cases,Lrecovery_cases,Ldeath_cases])

plt.show()
print(f'The total number of CONFIRMED Cases during LOCKDOWN is :{(Lconfrimed_cases)}')

print(f'The total number of ACTIVE Cases during LOCKDOWN is :{(Lactive_cases)}')

print(f'The total number of RECOVERY Cases during LOCKDOWN is :{(Lrecovery_cases)}')

print(f'The total number of DEATH Cases during LOCKDOWN is :{(Ldeath_cases)}')
After_Lockdown=new_data["20/4/2020":"20/8/2020"]

After_Lockdown.head()

Aconfrimed_cases=After_Lockdown["confirmed"].sum()

Aactive_cases=(After_Lockdown["confirmed"].sum()-After_Lockdown["recovered"].sum())-After_Lockdown["death"].sum()

Arecovery_cases=After_Lockdown["recovered"].sum()

Adeath_cases=After_Lockdown["death"].sum()
plt.figure(figsize=(20,10))

plt.title("AFTER LOCKDOWN PERIOD")

plt.xlabel("CASES")

plt.ylabel("Number_Frequency")

sns.barplot(x=["CONFIRMED","ACTIVE","RECOVERY","DEATH"],y=[Aconfrimed_cases,Aactive_cases,Arecovery_cases,Adeath_cases])

plt.show()
print(f'The total number of CONFIRMED Cases AFTER LOCKDOWN is :{(Aconfrimed_cases)}')

print(f'The total number of ACTIVE Cases AFTER LOCKDOWN is :{(Aactive_cases)}')

print(f'The total number of RECOVERY Cases AFTER LOCKDOWN is :{(Arecovery_cases)}')

print(f'The total number of DEATH Cases AFTER LOCKDOWN is :{(Adeath_cases)}')
plt.figure(figsize=(10,10))

plt.pie([Aconfrimed_cases,Aactive_cases,Arecovery_cases,Adeath_cases],labels=["CONFIRMED","ACTIVE","RECOVERY","DEATH"],

        autopct='%1.1f%%')

plt.show()
filename="../input/ghana-covid19-data/regional cases.xlsx"

region=pd.read_excel(filename)

region.head()





region['Percentage']=round((region['CUMILATIVE COUNTS']/ region['POPULATION']*100)*100,2)

region
last_update="202/08/20"

plt.figure(figsize=(15,5))

sns.barplot(x=region['REGION'],y=region['CUMILATIVE COUNTS'])

plt.title('Regional Active Cases (as of {0})'.format(last_update))

plt.xlabel("Region")

plt.ylabel("Cumilative Counts")

plt.xticks(rotation=70)

plt.show
plt.figure(figsize=(10,10))

sns.heatmap(new_data.corr(),annot=True,square=True,linewidths=.5,cbar_kws={'shrink':.5},center=0)

plt.show()
afterRevision=new_data["13/3/2020":"17/6/2020"]

afterRevision.head()
plt.figure(figsize=(10,10))

sns.heatmap(afterRevision.corr(),annot=True,square=True,linewidths=.5,cbar_kws={'shrink':.5},center=0)

plt.show()
sns.pairplot(new_data)

plt.show()
new_data.columns


plt.figure(figsize=(20,15))

sns.lmplot(x='cumulative_confirmed ',y='cumulative_death',data=new_data)

plt.xlabel('cumulative_confirmed')

plt.ylabel('cumulative_death')

plt.title('cumulative_confirmed vs cumulative_death')

plt.show()
newfile="../input/new-ghana-covid19-cases/Ghana_Covid19_DailyActive.csv"

model_data=pd.read_csv(newfile)

model_data.head()
model_data.columns
new_df = model_data.drop(['death','confirmed','recovered','date'],axis='columns')

new_df.head()
new_df.info()
# new_df['date'] =  pd.to_datetime(new_df['date'])



# new_df['date_delta'] = (new_df['date'] - new_df['date'].min())  / np.timedelta64(1,'D')
sns.pairplot(new_df)

plt.show()
X = new_df[['cumulative_confirmed ','cumulative_recovered','cumulative_death']]
y = new_df['active_cases']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
X_train.head()
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
model.predict(X_test)
y_test
model.score(X_test, y_test)