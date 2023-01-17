import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

%matplotlib inline
df1=pd.read_csv('../input/covid19/test (1).csv')

df2=pd.read_csv('../input/covid19/train (1).csv')

df3=pd.read_csv('../input/covid19/submission.csv')
df1.shape
df2.shape
df2.head(10)
df2.columns
df2.describe()
df2.columns = ['Id', 'Province_State', 'Country', 'Date', 'Confirmed_Cases',

       'Fatalities']
df2.columns
df2.isnull().sum()
sns.heatmap(df2.isnull())
df2.drop('Province_State', axis=1,inplace=True)
df2.head(3)
a=df2['Country'].value_counts()[:20]

b=a.index

a
b
covid_countries = list(b)

type(covid_countries)
conf_us=df2.loc[df2['Country'] == 'Kenya', 'Confirmed_Cases'].sum()

death_us=df2.loc[df2['Country'] == 'Kenya', 'Fatalities'].sum()

print(conf_us)

print(death_us)
d_lis=[]

c_lis2=[]

for i in covid_countries:

    

    conf_=df2.loc[df2['Country'] == i, 'Confirmed_Cases'].sum()

    death_=df2.loc[df2['Country'] == i, 'Fatalities'].sum()

    

    d_lis.append(int(death_))

    c_lis2.append(int(conf_))

c_lis2
plt.figure(figsize=(10,7))

for i,v in enumerate(a):

    plt.text(i,v,v, color='blue', fontweight='bold')

sns.barplot(x=a.index,y=a,orient='v')

plt.xticks(rotation=90,fontsize=12,)

plt.title('Top 20 effected Countries',fontsize=20)

plt.xlabel('Countries',fontsize=15)

plt.ylabel('Total No. of cases of COVID-19',fontsize=14)
data = {'Country':['US', 'China', 'France', 'Canada', 'Australia', 'United Kingdom',

       'Netherlands', 'Denmark', 'Armenia', 'Monaco', 'Cyprus', 'Poland',

       'Haiti', 'Ecuador', 'Central African Republic', 'Oman', 'Uganda',

       'Austria', 'Djibouti', 'El Salvador'], 'Cases':[3780,2310,700,700,560,490,280,210,70,70,70,70,70,70,70,70,70,70,70,70],

        'Casualties':c_lis2, 'Death':d_lis}

df = pd.DataFrame(data)

df
df.columns
x = df[['Casualties', 'Death']]
x.plot()
plt.figure(figsize=(10,7))

ax = df.plot(x='Country',y='Cases', kind='bar', color='black')

df.plot(x="Country", y="Death", kind="bar", ax=ax, color='r' )

df.plot(x="Country", y="Casualties", kind="bar", ax=ax, color='g')

plt.xticks(rotation=90,fontsize=12)
plt.figure(figsize=(10,7))

ax = df.plot(x='Country',y=['Casualties','Death'], kind='bar',width=1)
df
df2.head()
#US

ConfirmedCases_date_US = df2[df2['Country']=='US'].groupby(['Date']).agg({'Confirmed_Cases':['sum']})

fatalities_date_US = df2[df2['Country']=='US'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_US = ConfirmedCases_date_US.join(fatalities_date_US)





#China

ConfirmedCases_date_China = df2[df2['Country']=='China'].groupby(['Date']).agg({'Confirmed_Cases':['sum']})

fatalities_date_China = df2[df2['Country']=='China'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_China = ConfirmedCases_date_China.join(fatalities_date_China)



#France

ConfirmedCases_date_France = df2[df2['Country']=='France'].groupby(['Date']).agg({'Confirmed_Cases':['sum']})

fatalities_date_France = df2[df2['Country']=='France'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_France = ConfirmedCases_date_France.join(fatalities_date_France)

#Australia

ConfirmedCases_date_Australia = df2[df2['Country']=='Australia'].groupby(['Date']).agg({'Confirmed_Cases':['sum']})

fatalities_date_Australia = df2[df2['Country']=='Australia'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Australia = ConfirmedCases_date_Australia.join(fatalities_date_Australia)







plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_US.plot(ax=plt.gca(), title='US')

plt.ylabel("Confirmed  cases", size=13)



plt.subplot(2, 2, 2)

total_date_China.plot(ax=plt.gca(), title='China')



plt.subplot(2, 2, 3)

total_date_France.plot(ax=plt.gca(), title='France')

plt.ylabel("Confirmed cases", size=13)



plt.subplot(2, 2, 4)

total_date_Australia.plot(ax=plt.gca(), title='Australia')
#Canada

ConfirmedCases_date_Canada = df2[df2['Country']=='Canada'].groupby(['Date']).agg({'Confirmed_Cases':['sum']})

fatalities_date_Canada = df2[df2['Country']=='Canada'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Canada = ConfirmedCases_date_Canada.join(fatalities_date_Canada)





#Netherlands

ConfirmedCases_date_Netherlands = df2[df2['Country']=='Netherlands'].groupby(['Date']).agg({'Confirmed_Cases':['sum']})

fatalities_date_Netherlands = df2[df2['Country']=='Netherlands'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Netherlands = ConfirmedCases_date_Netherlands.join(fatalities_date_Netherlands)



#Denmark

ConfirmedCases_date_Denmark = df2[df2['Country']=='Denmark'].groupby(['Date']).agg({'Confirmed_Cases':['sum']})

fatalities_date_Denmark = df2[df2['Country']=='Denmark'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Denmark = ConfirmedCases_date_Denmark.join(fatalities_date_Denmark)

#Australia

ConfirmedCases_date_Cyprus = df2[df2['Country']=='Cyprus'].groupby(['Date']).agg({'Confirmed_Cases':['sum']})

fatalities_date_Cyprus = df2[df2['Country']=='Cyprus'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Cyprus = ConfirmedCases_date_Cyprus.join(fatalities_date_Cyprus)







plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_Canada.plot(ax=plt.gca(), title='Canada')

plt.ylabel("Confirmed  cases", size=13)



plt.subplot(2, 2, 2)

total_date_Netherlands.plot(ax=plt.gca(), title='Netherlands')



plt.subplot(2, 2, 3)

total_date_Denmark.plot(ax=plt.gca(), title='Denmark')

plt.ylabel("Confirmed cases", size=13)



plt.subplot(2, 2, 4)

total_date_Cyprus.plot(ax=plt.gca(), title='Cyprus')
fig = plt.figure(figsize=(16,8))

ax = fig.add_subplot(111)

df2.groupby('Country').mean().sort_values(by='Confirmed_Cases',

        ascending=False)['Confirmed_Cases'].plot('bar', color='r',width=0.3,title='Country Region Confirmed Cases',

                                                                                    fontsize=10)

plt.xticks(rotation = 90, fontsize=6)

plt.ylabel('Confirmed Cases')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(10)

ax.yaxis.label.set_fontsize(10)