import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
sns.set()
%matplotlib inline
data = pd.read_csv('../input/Attacks on Political Leaders in Pakistan.csv', encoding='latin1')
df = data.copy()
df.head()
df.info()
df.describe(include='all')
df.isnull().sum()
df['Location Category'].unique()
df['Location Category'].fillna('UNKNOWN', inplace=True)
df['Date'].unique()
df['Date'] = df['Date'].str.replace('16-Oct-51', '16-Oct-1951')
df['Date'] = df['Date'].str.replace('9-May-58', '9-Oct-1958')
df['Date'] = pd.to_datetime(df['Date'])
df['Politician'].unique()
df['Politician'] = df['Politician'].str.strip()
df['Politician'] = df['Politician'].str.replace('\xa0Asim Ali Kurd', 'Asim Ali Kurd')
df['Politician'] = df['Politician'].str.replace('\xa0Amjad Ali Khan', 'Amjad Ali Khan')
df['Politician'] = df['Politician'].str.replace('\xa0Khalid Mumtaz Kundi', 'Khalid Mumtaz Kundi')
df['Politician'] = df['Politician'].str.replace('Dr Mohammad Ibrahim Jatoi\xa0', 'Dr Mohammad Ibrahim Jatoi')
df['Politician'] = df['Politician'].str.replace('Col Shuja Khanzada\xa0', 'Col Shuja Khanzada')
df['City'].unique()
df['City'] = df['City'].str.strip()
df['City'] = df['City'].str.replace('ATTOCK', 'Attock')
df['City'] = df['City'].str.replace('KALAT', 'Kalat')
df['Location'].unique()
df['Location'] = df['Location'].str.strip()
df['Location'] = df['Location'].str.replace('\xa0Sadullah Khan\'s house at 16 Aikman Road', 'Sadullah Khan\'s house at 16 Aikman Road')
df['Location'] = df['Location'].str.replace('\xa0Zarghon Road 200 metres from Chief Minister\x92s House', 'Zarghon Road 200 metres from Chief Minister\'s House')
df['Location'] = df['Location'].str.replace('village of Palaseen\nNear Finance Minnister\nHouse Quetta', 'village of Palaseen Near Finance Minnister House Quetta')
df['Location'] = df['Location'].str.replace('Ghalani area\nAgent office ', 'Ghalani area Agent office')
df['Location'] = df['Location'].str.replace('highly contested partial rerun of the\xa0general election', 'highly contested partial rerun of the general election')
df['Province'].unique()
df['Province'] = df['Province'].str.replace('FATA', 'Fata')
df['Party'].unique()
df['Party'] = df['Party'].str.replace('Alll India Muslim League', 'All India Muslim League')
df['Party'] = df['Party'].str.replace('Hazara Democratic Party\xa0(HDP)', 'Hazara Democratic Party(HDP)')
df.drop('S#', inplace=True, axis=1)
df['Killed'].unique()
plt.figure(figsize=(15, 5))
plt.subplot(121)
sns.distplot(df['Killed']);
plt.subplot(122)
sns.distplot(df['Injured']);
plt.figure(figsize=(20, 5))
plt.subplot(121)
sns.barplot(x='Killed', y='Day', data=df, ci=None);
plt.subplot(122)
sns.barplot(x='Injured', y='Day', data=df, ci=None);
plt.figure(figsize=(20, 5))
plt.subplot(121)
sns.barplot(y='Killed', x='Day Type', data=df, ci=None);
plt.subplot(122)
sns.barplot(y='Injured', x='Day Type', data=df, ci=None);
plt.figure(figsize=(20, 5))
plt.subplot(121)
sns.barplot(x='Killed', y='Time', data=df, ci=None);
plt.subplot(122)
sns.barplot(x='Injured', y='Time', data=df, ci=None);
plt.figure(figsize=(20, 10))
plt.subplot(121)
sns.barplot(x='Killed', y='City', data=df, ci=None);
plt.subplot(122)
sns.barplot(x='Injured', y='City', data=df, ci=None);
plt.figure(figsize=(20, 5))
plt.subplot(121)
sns.barplot(x='Killed', y='Province', data=df, ci=None);
plt.subplot(122)
sns.barplot(x='Injured', y='Province', data=df, ci=None);
plt.figure(figsize=(20, 5))
plt.subplot(121)
sns.barplot(y='Killed', x='Target Category', data=df, ci=None);
plt.subplot(122)
sns.barplot(y='Injured', x='Target Category', data=df, ci=None);
plt.figure(figsize=(20, 5))
plt.subplot(121)
sns.barplot(y='Killed', x='Space (Open/Closed)', data=df, ci=None);
plt.subplot(122)
sns.barplot(y='Injured', x='Space (Open/Closed)', data=df, ci=None);
m = folium.Map([30.3753, 69.3451], zoom_start=5.4, tiles="Mapbox Bright")
for i in range(0, len(df)):
    folium.Marker([df.iloc[i]['Latitude'], df.iloc[i]['Longititude']], popup=df.iloc[i]['Politician']).add_to(m)
m
plt.figure(figsize=(20, 10))
sns.barplot(x='Killed', y='Party', data=df, ci=None);
plt.figure(figsize=(20, 10))
sns.barplot(x='Injured', y='Party', data=df, ci=None);
plt.figure(figsize=(20, 15))
sns.countplot(y='Politician', hue='Target Status', data=df);
plt.figure(figsize=(20, 15))
sns.countplot(y='Politician', hue='Day Type', data=df);
plt.figure(figsize=(20, 15))
sns.countplot(y='Politician', hue='Target Category', data=df);
sns.pairplot(df, hue="Killed", height=3.5)
sns.pairplot(df, hue="Injured", height=3.5)