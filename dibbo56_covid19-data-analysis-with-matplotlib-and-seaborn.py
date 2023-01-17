import datetime

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter,DayLocator

import matplotlib.dates as mdates

import matplotlib.style

import seaborn as sns

sns.set(style="white")

matplotlib.style.use('ggplot')





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('../input/covid19/train_data.csv')

df.head()
print('Data information')

print(df.info(), end='\n\n')

print('Checking for null values')

print(df.isnull().sum(), end='\n\n')

print('Necessary information from the dataset')

print('Total affected countries ', len(df['Country_Region'].unique()))

print('Total confirmed cases ', df['ConfirmedCases'].sum())

print('Total fatalities cases ', df['Fatalities'].sum())

d = df['Date'].unique()

date = {}

for i in d:

    date.update({i:0})

    

for i in date:

    date.update({i:df[df['Date']==i]['ConfirmedCases'].sum()})
x_values = [datetime.datetime.strptime(d,"%Y-%m-%d").date() for d in date.keys()]

y_values = date.values()
plt.figure(figsize=(10,6))

x_values = [i for i in range(1,82)]

y_values = [i for i in y_values]

ax = sns.lineplot(x=x_values, y=y_values, color='coral', label="line")

ax.set_title('Global Confirmed Cases')

ax.set(xlabel="Date in X axis", ylabel = "Confirmed Cases in Y axis")

plt.show()
d = df['Date'].unique()

date = {}

for i in d:

    date.update({i:0})

    

for i in date:

    date.update({i:df[df['Date']==i]['Fatalities'].sum()})
x_values = [datetime.datetime.strptime(d,"%Y-%m-%d").date() for d in date.keys()]

y_values = date.values()
plt.figure(figsize=(10,6))

x_values = [i for i in range(1,82)]

y_values = [i for i in y_values]

ax = sns.lineplot(x=x_values, y=y_values, color='coral', label="line")

ax.set_title('Global Fatality Cases')

ax.set(xlabel="Date in X axis", ylabel = "Fatality Cases in Y axis")

plt.show()
data = {'Country':[], 'ConfirmedCases':[], 'Fatalities':[]}

data.update({'Country':df['Country_Region'].unique()})



confirm_case = []

for i in data['Country']:

    confirm_case.append(df[df['Country_Region'] == i]['ConfirmedCases'].sum())



fatalities_case = []

for i in data['Country']:

    fatalities_case.append(df[df['Country_Region'] == i]['Fatalities'].sum())

    

data.update({'ConfirmedCases':confirm_case})

data.update({'Fatalities':fatalities_case})
print(len(data['Country']))

print(len(data['ConfirmedCases']))

print(len(data['Fatalities']))
data = pd.DataFrame(data)

data.head()
df_confirm_asc = data.sort_values(by=['ConfirmedCases'], ascending=False)
df_confirm_asc = df_confirm_asc.reset_index(drop=True)

df_confirm_asc.style.background_gradient(cmap="Reds")
x_values = [i for i in df_confirm_asc.loc[0:9,'Country']]

y_values = [i for i in df_confirm_asc.loc[0:9,'ConfirmedCases']]
plt.figure(figsize=(10,6))

ax = sns.barplot(x=y_values, y=x_values, palette='Reds_r')

ax.set_title('Top 10 Highest Confirmed Cases Country')

ax.set(xlabel="Confirmed Cases in X axis", ylabel = "Countries in Y axis")

plt.show()
df_fatality_asc = data.sort_values(by=['Fatalities'], ascending=False)

df_fatality_asc = df_fatality_asc.reset_index(drop=True)

df_fatality_asc.style.background_gradient(cmap="Reds")
x_values = [i for i in df_fatality_asc.loc[0:9,'Country']]

y_values = [i for i in df_fatality_asc.loc[0:9,'Fatalities']]
plt.figure(figsize=(10,6))

ax = sns.barplot(x=y_values, y=x_values, palette='Reds_r')

ax.set_title('Top 10 Highest Fatality Cases Country')

ax.set(xlabel="Fatality Cases in X axis", ylabel = "Countries in Y axis")

plt.show()
df_confirm_asc = data.sort_values(by=['ConfirmedCases'], ascending=True)

df_confirm_asc = df_confirm_asc.reset_index(drop=True)

df_confirm_asc.style.background_gradient(cmap="Reds")
x_values = [i for i in df_confirm_asc.loc[0:9,'Country']]

y_values = [i for i in df_confirm_asc.loc[0:9,'ConfirmedCases']]
plt.figure(figsize=(10,6))

ax = sns.barplot(x=y_values, y=x_values, palette='Reds')

ax.set_title('10 Lowest Confirmed Cases Country')

ax.set(xlabel="Confirmed Cases in X axis", ylabel = "Countries in Y axis")

plt.show()
df_fatality_asc = data.sort_values(by=['Fatalities'], ascending=True)

df_fatality_asc = df_fatality_asc.reset_index(drop=True)

df_fatality_asc.style.background_gradient(cmap="Reds")
x_values = [i for i in df_fatality_asc.loc[34:43,'Country']]

y_values = [i for i in df_fatality_asc.loc[34:43,'Fatalities']]
plt.figure(figsize=(10,6))

ax = sns.barplot(x=y_values, y=x_values, palette='Reds')

ax.set_title('10 Lowest Fatality Cases Country')

ax.set(xlabel="Fatality Cases in X axis", ylabel = "Countries in Y axis")

plt.show()
df_confirm_asc = data.sort_values(by=['ConfirmedCases'], ascending=False)

df_confirm_asc = df_confirm_asc.reset_index(drop=True)

x_values = [i for i in df_confirm_asc.loc[0:9,'Country']]

y_values = [i for i in df_confirm_asc.loc[0:9,'ConfirmedCases']]


plt.figure(figsize=(5,5))

plt.axis('equal')

plt.pie(y_values, labels=x_values, radius=2, autopct='%.2f%%', shadow=True, startangle=0)

plt.title('Top 10 Confirmed Cases Country')

plt.show()
df_fatality_asc = data.sort_values(by=['Fatalities'], ascending=False)

df_fatality_asc = df_fatality_asc.reset_index(drop=True)

x_values = [i for i in df_fatality_asc.loc[0:9,'Country']]

y_values = [i for i in df_fatality_asc.loc[0:9,'Fatalities']]


plt.figure(figsize=(5,5))

plt.axis('equal')

plt.pie(y_values, labels=x_values, radius=2, autopct='%.2f%%', shadow=True, startangle=0)

plt.title('Top 10 Fatality Cases Country')

plt.show()
data = {'Date':[], 'US':[], 'China':[], 'Italy':[], 'Spain':[], 'Germany':[], 'France':[], 'Iran':[]}

data.update({'Date': df['Date'].unique()})



for i in data['Date']:

    data['US'].append(df[(df['Date']==i) & (df['Country_Region']=='US')]['ConfirmedCases'].sum())

    data['China'].append(df[(df['Date']==i) & (df['Country_Region']=='China')]['ConfirmedCases'].sum())

    data['Italy'].append(df[(df['Date']==i) & (df['Country_Region']=='Italy')]['ConfirmedCases'].sum())

    data['Spain'].append(df[(df['Date']==i) & (df['Country_Region']=='Spain')]['ConfirmedCases'].sum())

    data['Germany'].append(df[(df['Date']==i) & (df['Country_Region']=='Germany')]['ConfirmedCases'].sum())

    data['France'].append(df[(df['Date']==i) & (df['Country_Region']=='France')]['ConfirmedCases'].sum())

    data['Iran'].append(df[(df['Date']==i) & (df['Country_Region']=='Iran')]['ConfirmedCases'].sum())

    

data = pd.DataFrame(data)

data.head()  
ax = plt.figure(figsize=(10,6))

a = sns.lineplot(x=data.index, y=data.US, data = data)

a = sns.lineplot(x=data.index, y=data.China, data = data )

a = sns.lineplot(x=data.index, y=data.Italy, data = data )

a = sns.lineplot(x=data.index, y=data.Spain, data = data )

a = sns.lineplot(x=data.index, y=data.Germany, data = data )

a = sns.lineplot(x=data.index, y=data.France, data = data )

a = sns.lineplot(x=data.index, y=data['Iran'], data = data )

ax.legend(['US', 'China', 'Italy', 'Spain', 'Germany', 'France', 'Iran'], loc='upper right')

a.set_title('US, Chian, Italy, Spain, Germany, France, Iran Confirmed Cases \nFrom 2020/01/22 to 2020/04/11')

a.set(xlabel="Date in X axis", ylabel = "Confirmed Cases in Y axis")

plt.show()
data = {'Date':[], 'US':[], 'China':[], 'Italy':[], 'Spain':[], 'Germany':[], 'France':[], 'Iran':[]}

data.update({'Date': df['Date'].unique()})



for i in data['Date']:

    data['US'].append(df[(df['Date']==i) & (df['Country_Region']=='US')]['Fatalities'].sum())

    data['China'].append(df[(df['Date']==i) & (df['Country_Region']=='China')]['Fatalities'].sum())

    data['Italy'].append(df[(df['Date']==i) & (df['Country_Region']=='Italy')]['Fatalities'].sum())

    data['Spain'].append(df[(df['Date']==i) & (df['Country_Region']=='Spain')]['Fatalities'].sum())

    data['Germany'].append(df[(df['Date']==i) & (df['Country_Region']=='Germany')]['Fatalities'].sum())

    data['France'].append(df[(df['Date']==i) & (df['Country_Region']=='France')]['Fatalities'].sum())

    data['Iran'].append(df[(df['Date']==i) & (df['Country_Region']=='Iran')]['Fatalities'].sum())

    

data = pd.DataFrame(data)

data.head()  
ax = plt.figure(figsize=(10,6))

a = sns.lineplot(x=data.index, y=data.US, data = data)

a = sns.lineplot(x=data.index, y=data.China, data = data )

a = sns.lineplot(x=data.index, y=data.Italy, data = data )

a = sns.lineplot(x=data.index, y=data.Spain, data = data )

a = sns.lineplot(x=data.index, y=data.Germany, data = data )

a = sns.lineplot(x=data.index, y=data.France, data = data )

a = sns.lineplot(x=data.index, y=data['Iran'], data = data )

ax.legend(['US', 'China', 'Italy', 'Spain', 'Germany', 'France', 'Iran'], loc='upper right')

a.set_title('US, Chian, Italy, Spain, Germany, France, Iran Fatality Cases \nFrom 2020/01/22 to 2020/04/11')

a.set(xlabel="Date in X axis", ylabel = "Fatality Cases in Y axis")

plt.show()