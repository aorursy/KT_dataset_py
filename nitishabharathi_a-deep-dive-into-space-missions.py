import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

#plt.style.use('dark_background')

data = pd.read_csv('../input/all-space-missions-from-1957/Space_Corrected.csv')



# Data Cleaning

data = data.drop(['Unnamed: 0','Unnamed: 0.1'],1)

data['Country'] = data['Location'].apply(lambda x:x.split(',')[-1])

data['year'] = data['Datum'].apply(lambda x:x.split()[3])
status = list(data['Status Mission'].value_counts().keys())

status_values = list(data['Status Mission'].value_counts())



colors = ['#17AF4A','#EF473C','#F78980','#250704']



plt.figure(figsize= (8,8))

plt.title('Status of Mission',fontsize = 20)

plt.pie(status_values, labels=status, colors=colors,autopct='%1.1f%%')

plt.axis('equal')

plt.tight_layout()
company_name = list(data['Company Name'].value_counts().keys())

values = list(data['Company Name'].value_counts())

company_name = company_name[:15]

values = values[:15]

plt.figure(figsize=(8,8))

sns.set_color_codes("pastel")

plt.title('Top 15 Space Companies and their launches', fontsize = 20)

sns.barplot(x= values, y= company_name,color = '#9370db');
status = list(data['Status Rocket'].value_counts().keys())

status_values = list(data['Status Rocket'].value_counts())



explode = (0, 0.1)

colors = ['#FC818E','#66b3ff']



plt.figure(figsize= (8,8))

plt.title('Status of Rockets',fontsize = 20)

plt.pie(status_values, labels=status, colors=colors, autopct='%1.1f%%',shadow=True, startangle=90,explode = explode)

plt.axis('equal')

plt.tight_layout()
active = data[data['Status Rocket'] == 'StatusActive']

retired = data[data['Status Rocket'] == 'StatusRetired']

active_value = []

retired_value = []

for i in company_name:

    df1 = active[active['Company Name'] == i]

    active_value.append(len(df1))

    df2 = retired[retired['Company Name'] == i]

    retired_value.append(len(df2))

plt.figure(figsize=(10,8))

sns.barplot(x = retired_value, y= company_name,label="Retired Rockets", color = '#3FE2D8')

sns.barplot(x = active_value, y= company_name,label="Active Rockets", color = '#FD702F')

plt.title('Company wise Rocket Status',fontsize = 20)

plt.legend(ncol=2, loc="lower right", frameon=True);
countries = list(data['Country'].value_counts().keys())

values = list(data['Country'].value_counts())

plt.figure(figsize=(15,8))

sns.set_color_codes("pastel")

plt.title('Country wise launches', fontsize = 20)

plt.xticks(rotation = 90 ,fontsize = 11)

sns.barplot(x= countries, y= values,color = '#CE596F');
sns.set_palette("husl")



year = list(data['year'].value_counts().keys())

values = list(data['year'].value_counts())

plt.figure(figsize=(15,8))

sns.set_color_codes("pastel")

plt.title('Year wise launches', fontsize = 20)

plt.xticks(rotation = 90 ,fontsize = 11)

sns.barplot(x= year, y= values);
data[' Rocket'].isna().value_counts()
data[' Rocket']=data[' Rocket'].str.replace(',','')

data[' Rocket']=data[' Rocket'].astype(float)



cost = data.dropna()

plt.figure(figsize=(10,8))

sns.distplot(cost[' Rocket']);

plt.title('Mission cost')

plt.xlabel('Cost of mission',size=15)