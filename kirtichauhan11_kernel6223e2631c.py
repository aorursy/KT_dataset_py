import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('../input/uber-data/Uber Request Data1.csv')
df.head()
df.shape
df.columns
df.drop_duplicates('Request id')
df.isnull().sum()
print(100*(df.isnull().sum())/6745)
df.describe
df.info()
df['Request timestamp']=df['Request timestamp'].astype(str)

df['Request timestamp']=df['Request timestamp'].str.replace('/','-')

df['Request timestamp']=pd.to_datetime(df['Request timestamp'],dayfirst=True)
df['Drop timestamp']=df['Drop timestamp'].astype(str)

df['Drop timestamp']=df['Drop timestamp'].str.replace('/','-')

df['Drop timestamp']=pd.to_datetime(df['Drop timestamp'],dayfirst=True)
df.info()
df
day=df['Request timestamp'].dt.day

day.head()

print(day.value_counts())

df['day']=day
hour=df['Request timestamp'].dt.hour

hour.head()

print(hour.value_counts())

df['hour']=hour
sns.factorplot(x ='hour',hue='Status',row='day', data = df,kind='count')

plt.show()
df.columns
sns.factorplot(x='hour', hue='Pickup point',data=df, kind='count')
def timespan(hr):

    if hr<5:

        return 'Pre Morning'

    elif 5<hr<10:

        return 'Morning'

    elif 10<hr<17:

        return 'After 10am'

    elif 17<hr<22:

        return 'Eve'

    else:

        return 'Night'

df['timeslot']=df.hour.apply(lambda hr: timespan(hr))
df['timeslot'].value_counts()
sns.factorplot(x='timeslot', hue='Pickup point' , data=df , kind='count')
sns.factorplot(x='timeslot', hue='Status',row ='Pickup point', data= df , kind='count')
df_mor=df[df.timeslot=='Morning']

sns.countplot(x='Pickup point',hue='Status',data=df_mor)
df_city_cars_cancelled=df_mor.loc[(df_mor['Pickup point']=='City') & (df_mor['Status']=='Cancelled')]

print(len(df_city_cars_cancelled))
#lets see demand and supply in city in the morning slot

df_city_comp=df_mor.loc[(df_mor['Pickup point']=='City') & (df_mor['Status']=='Trip Completed')]

print(len(df_city_comp))
df_city_demand=df_mor.loc[df_mor['Pickup point']=='City']

print(len(df_city_demand))
df_airport_cars_cancelled=df_mor.loc[(df_mor['Pickup point']=='Airport') & (df_mor['Status']=='Cancelled')]

print(len(df_airport_cars_cancelled))
#lets see demand and supply in airport in the morning slot

df_airport_comp=df_mor.loc[(df_mor['Pickup point']=='Airport')  & (df_mor['Status']=='Trip Completed')]

print(len(df_airport_comp))
df_airport_demand=df_mor.loc[df_mor['Pickup point']=='Airport']

print(len(df_airport_demand))
df_eve=df[df.timeslot=='Eve']

sns.countplot(x='Pickup point', hue='Status', data= df_eve)
df_city_cars_notavailable=df_eve.loc[(df_eve['Pickup point']=='City') & (df_eve['Status']=='No Cars Available')]

print(len(df_city_cars_notavailable))
#lets see demand and supply in city in the evening slot

df_city_completed=df_eve.loc[(df_eve['Pickup point']=='City') & (df_eve['Status']=='Trip Completed')]

print(len(df_city_completed))
df_city_cars_demand=df_eve.loc[df_eve['Pickup point']=='City']

print(len(df_city_cars_demand))
df_airport_cars_notavailable=df_eve.loc[(df_eve['Pickup point']=='Airport') & (df_eve['Status']=='No Cars Available')]

print(len(df_airport_cars_notavailable))
#lets see demand and supply in airport in the evening slot 

df_airport_completed=df_eve.loc[(df_eve['Pickup point']=='Airport') & (df_eve['Status']=='Trip Completed')]

print(len(df_airport_completed))
df_airport_cars_demand=df_eve.loc[df_eve['Pickup point']=='Airport']

print(len(df_airport_cars_demand))
df_city=df.loc[(df['Pickup point']=='City') & (df['timeslot']=='Morning')]

df_city_plot=pd.DataFrame(df_city.Status.value_counts())

labels=df_city_plot.index.values

plt.pie(df_city_plot,labels=labels,autopct='%1.1f%%')
df_airport=df.loc[(df['Pickup point']=='Airport') & (df['timeslot']=='Eve')]

df_airport_plot=pd.DataFrame(df_airport.Status.value_counts())

labels=df_airport_plot.index.values

plt.pie(df_airport_plot,labels=labels,autopct='%1.1f%%')
