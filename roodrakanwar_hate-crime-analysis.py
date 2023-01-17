import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
cbd = pd.read_csv("../input/crimeanalysis/crime_by_district.csv")

cbdr = pd.read_csv("../input/crimeanalysis/crime_by_district_rt.csv")

cbs = pd.read_csv("../input/crimeanalysis/crime_by_state.csv")

cbsr = pd.read_csv("../input/crimeanalysis/crime_by_state_rt.csv")
cbd.head()
cbdr.head()
cbd.shape
cbdr.shape
total1 = pd.concat([cbd,cbdr]).drop_duplicates(keep = False)
total1['Total Atrocities'] = total1['Murder'] +total1['Assault on women']+total1['Kidnapping and Abduction']+total1['Dacoity']+total1['Robbery']+total1['Arson']+total1['Hurt']+total1['Prevention of atrocities (POA) Act']+total1['Protection of Civil Rights (PCR) Act']+total1['Other Crimes Against SCs']

total1.head()
cbs.shape
cbsr.shape
total = pd.concat([cbs,cbsr]).drop_duplicates(keep = False)
total.shape
total.head(20)
total.tail(20)
total.drop(total[total['STATE/UT'] == 'TOTAL (UTs)'].index , inplace = True) 

total.drop(total[total['STATE/UT'] == 'TOTAL (STATES)'].index , inplace = True) 
cbdr.isnull().sum()
cbsr.isnull().sum()
total.isnull().sum()
cbdr['Total Atrocities'] = cbdr['Murder'] +cbdr['Assault on women']+cbdr['Kidnapping and Abduction']+cbdr['Dacoity']+cbdr['Robbery']+cbdr['Arson']+cbdr['Hurt']+cbdr['Prevention of atrocities (POA) Act']+cbdr['Protection of Civil Rights (PCR) Act']+cbdr['Other Crimes Against SCs']

s= cbdr.groupby(['STATE/UT','DISTRICT','Year'])['Murder'].sum().reset_index().sort_values(by='Murder',ascending=False)

s.head(10).style.background_gradient(cmap='Reds')
s= cbdr.groupby(['STATE/UT','DISTRICT','Year'])['Assault on women'].sum().reset_index().sort_values(by='Assault on women',ascending=False)

s.head(10).style.background_gradient(cmap='Purples')
s= cbdr.groupby(['STATE/UT','DISTRICT','Year'])['Kidnapping and Abduction'].sum().reset_index().sort_values(by='Kidnapping and Abduction',ascending=False)

s.head(10).style.background_gradient(cmap='Blues')
s= cbdr.groupby(['STATE/UT','DISTRICT','Year'])['Dacoity'].sum().reset_index().sort_values(by='Dacoity',ascending=False)

s.head(10).style.background_gradient(cmap='Greens')
s= cbdr.groupby(['STATE/UT','DISTRICT','Year'])['Robbery'].sum().reset_index().sort_values(by='Robbery',ascending=False)

s.head(10).style.background_gradient(cmap='Oranges')
s= cbdr.groupby(['STATE/UT','DISTRICT','Year'])['Arson'].sum().reset_index().sort_values(by='Arson',ascending=False)

s.head(10).style.background_gradient(cmap='RdPu')
s= cbdr.groupby(['STATE/UT','DISTRICT','Year'])['Hurt'].sum().reset_index().sort_values(by='Hurt',ascending=False)

s.head(10).style.background_gradient(cmap='Greys')
s= cbdr.groupby(['STATE/UT','DISTRICT','Year'])['Prevention of atrocities (POA) Act'].sum().reset_index().sort_values(by='Prevention of atrocities (POA) Act',ascending=False)

s.head(10).style.background_gradient(cmap='Purples')
s= cbdr.groupby(['STATE/UT','DISTRICT','Year'])['Protection of Civil Rights (PCR) Act'].sum().reset_index().sort_values(by='Protection of Civil Rights (PCR) Act',ascending=False)

s.head(10).style.background_gradient(cmap='Greens')
s= cbdr.groupby(['STATE/UT','DISTRICT','Year'])['Other Crimes Against SCs'].sum().reset_index().sort_values(by='Other Crimes Against SCs',ascending=False)

s.head(10).style.background_gradient(cmap='Blues')
s= cbdr.groupby(['STATE/UT','DISTRICT','Year'])['Total Atrocities'].sum().reset_index().sort_values(by='Total Atrocities',ascending=False)

s.head(10).style.background_gradient(cmap='Greys')
sns.catplot(x='Year', y='Murder', data=cbdr,height = 5, aspect = 4)
sns.catplot(x='Year', y='Assault on women', data=cbdr,height = 5, aspect = 4)
sns.catplot(x='Year', y='Kidnapping and Abduction', data=cbdr,height = 5, aspect = 4)
sns.catplot(x='Year', y='Dacoity', data=cbdr,height = 5, aspect = 4)
sns.catplot(x='Year', y='Robbery', data=cbdr,height = 5, aspect = 4)
sns.catplot(x='Year', y='Arson', data=cbdr,height = 5, aspect = 4)
sns.catplot(x='Year', y='Hurt', data=cbdr,height = 5, aspect = 4)
sns.catplot(x='Year', y='Prevention of atrocities (POA) Act', data=cbdr,height = 5, aspect = 4)
sns.catplot(x='Year', y='Protection of Civil Rights (PCR) Act', data=cbdr,height = 5, aspect = 4)
sns.catplot(x='Year', y='Other Crimes Against SCs', data=cbdr,height = 5, aspect = 4)
sns.catplot(x='Year', y='Total Atrocities', data=cbdr,height = 5, aspect = 4)
cbsr['Total Atrocities'] = cbsr['Murder'] +cbsr['Assault on women']+cbsr['Kidnapping and Abduction']+cbsr['Dacoity']+cbsr['Robbery']+cbsr['Arson']+cbsr['Hurt']+cbsr['Prevention of atrocities (POA) Act']+cbsr['Protection of Civil Rights (PCR) Act']+cbsr['Other Crimes Against SCs']

cbsr.head()

sns.relplot(x ='Total Atrocities', y ='Year', col = 'STATE/UT', data = cbsr, height=3 ,col_wrap = 9)
s= cbsr.groupby(['STATE/UT','Year'])['Murder'].sum().reset_index().sort_values(by='Murder',ascending=False)

s.head(10).style.background_gradient(cmap='Reds')
s= cbsr.groupby(['STATE/UT','Year'])['Assault on women'].sum().reset_index().sort_values(by='Assault on women',ascending=False)

s.head(10).style.background_gradient(cmap='Purples')
s= cbsr.groupby(['STATE/UT','Year'])['Kidnapping and Abduction'].sum().reset_index().sort_values(by='Kidnapping and Abduction',ascending=False)

s.head(10).style.background_gradient(cmap='Blues')
s= cbsr.groupby(['STATE/UT','Year'])['Dacoity'].sum().reset_index().sort_values(by='Dacoity',ascending=False)

s.head(10).style.background_gradient(cmap='Greens')
s= cbsr.groupby(['STATE/UT','Year'])['Robbery'].sum().reset_index().sort_values(by='Robbery',ascending=False)

s.head(10).style.background_gradient(cmap='Oranges')
s= cbsr.groupby(['STATE/UT','Year'])['Arson'].sum().reset_index().sort_values(by='Arson',ascending=False)

s.head(10).style.background_gradient(cmap='RdPu')
s = cbsr.groupby(['STATE/UT','Year'])['Hurt'].sum().reset_index().sort_values(by='Hurt',ascending=False)

s.head(10).style.background_gradient(cmap='Greys')
s= cbsr.groupby(['STATE/UT','Year'])['Prevention of atrocities (POA) Act'].sum().reset_index().sort_values(by='Prevention of atrocities (POA) Act',ascending=False)

s.head(10).style.background_gradient(cmap='Purples')
s= cbsr.groupby(['STATE/UT','Year'])['Protection of Civil Rights (PCR) Act'].sum().reset_index().sort_values(by='Protection of Civil Rights (PCR) Act',ascending=False)

s.head(10).style.background_gradient(cmap='Greens')
s= cbsr.groupby(['STATE/UT','Year'])['Other Crimes Against SCs'].sum().reset_index().sort_values(by='Other Crimes Against SCs',ascending=False)

s.head(10).style.background_gradient(cmap='Blues')
s= cbsr.groupby(['STATE/UT','Year'])['Total Atrocities'].sum().reset_index().sort_values(by='Total Atrocities',ascending=False)

s.head(10).style.background_gradient(cmap='Greys')
x = cbsr['Year']

y = cbsr['Total Atrocities']
sns.axes_style('white')

sns.jointplot(x=x, y=y, kind = 'hex', color = 'green')
f, ax = plt.subplots(figsize=(6,6))

cmap = sns.cubehelix_palette(as_cmap = True, dark=0,light = 1,reverse=True)

sns.kdeplot(x,y,cmap=cmap, n_levels = 60, shade= True)
total['Total Atrocities'] = total['Murder'] +total['Assault on women']+total['Kidnapping and Abduction']+total['Dacoity']+total['Robbery']+total['Arson']+total['Hurt']+total['Prevention of atrocities (POA) Act']+total['Protection of Civil Rights (PCR) Act']+total['Other Crimes Against SCs']

total.head(15)
s= total.groupby(['Year'])['Murder'].sum().reset_index().sort_values(by='Murder',ascending=False)

s.head(15).style.background_gradient(cmap='Reds')
s= total.groupby(['Year'])['Assault on women'].sum().reset_index().sort_values(by='Assault on women',ascending=False)

s.head(15).style.background_gradient(cmap='Blues')
s= total.groupby(['Year'])['Kidnapping and Abduction'].sum().reset_index().sort_values(by='Kidnapping and Abduction',ascending=False)

s.head(12).style.background_gradient(cmap='Purples')
s= total.groupby(['Year'])['Dacoity'].sum().reset_index().sort_values(by='Dacoity',ascending=False)

s.head(15).style.background_gradient(cmap='Greens')
s= total.groupby(['Year'])['Robbery'].sum().reset_index().sort_values(by='Robbery',ascending=False)

s.head(15).style.background_gradient(cmap='Oranges')
s= total.groupby(['Year'])['Arson'].sum().reset_index().sort_values(by='Arson',ascending=False)

s.head(15).style.background_gradient(cmap='RdPu')
s = total.groupby(['Year'])['Hurt'].sum().reset_index().sort_values(by='Hurt',ascending=False)

s.head(15).style.background_gradient(cmap='Greys')
s= total.groupby(['Year'])['Prevention of atrocities (POA) Act'].sum().reset_index().sort_values(by='Prevention of atrocities (POA) Act',ascending=False)

s.head(15).style.background_gradient(cmap='Purples')
s= total.groupby(['Year'])['Protection of Civil Rights (PCR) Act'].sum().reset_index().sort_values(by='Protection of Civil Rights (PCR) Act',ascending=False)

s.head(15).style.background_gradient(cmap='Greens')
s= total.groupby(['Year'])['Other Crimes Against SCs'].sum().reset_index().sort_values(by='Other Crimes Against SCs',ascending=False)

s.head(15).style.background_gradient(cmap='Blues')
s= total.groupby(['Year'])['Total Atrocities'].sum().reset_index().sort_values(by='Total Atrocities',ascending=False)

s.head(15).style.background_gradient(cmap='Greys')
sns.catplot(x='Year', y='Murder', data=total ,height = 5, aspect = 4,kind = 'bar')
sns.catplot(x='Year', y='Assault on women', data=total ,height = 5, aspect = 4,kind = 'bar')
sns.catplot(x='Year', y='Kidnapping and Abduction', data=total ,height = 5, aspect = 4,kind = 'bar')
sns.catplot(x='Year', y='Dacoity', data=total ,height = 5, aspect = 4,kind = 'bar')
sns.catplot(x='Year', y='Robbery', data=total ,height = 5, aspect = 4,kind = 'bar')
sns.catplot(x='Year', y='Arson', data=total ,height = 5, aspect = 4,kind = 'bar')
sns.catplot(x='Year', y='Hurt', data=total ,height = 5, aspect = 4,kind = 'bar')
sns.catplot(x='Year', y='Prevention of atrocities (POA) Act', data=total ,height = 5, aspect = 4,kind = 'bar')
sns.catplot(x='Year', y='Protection of Civil Rights (PCR) Act', data=total ,height = 5, aspect = 4,kind = 'bar')
sns.catplot(x='Year', y='Other Crimes Against SCs', data=total ,height = 5, aspect = 4,kind = 'bar')
sns.catplot(x='Year', y='Total Atrocities', data=total ,height = 5, aspect = 4,kind = 'bar')