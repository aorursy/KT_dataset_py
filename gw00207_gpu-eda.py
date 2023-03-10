import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('../input/All_GPUs.csv')
#Convert release dates to useable format
df['Release_Date']=df['Release_Date'].str[1:-1]
df=df[df['Release_Date'].str.len()==11]
df['Release_Date']=pd.to_datetime(df['Release_Date'], format='%d-%b-%Y')
#Convert memory bandwidths to GB/s
s=df['Memory_Bandwidth']
s[(s.notnull()==True)&(s.str.contains('MB'))]=s[(s.notnull()==True)&(s.str.contains('MB'))].str[:-6].astype(int)/1000
s[(s.notnull()==True)&(s.str.contains('GB'))]=s[(s.notnull()==True)&(s.str.contains('GB'))].str[:-6].astype(float)
df['Memory_Bandwidth']=s
#Drop units from core_speed
df.Core_Speed=df.Core_Speed.str[:-4]
df.Core_Speed=df.Core_Speed.replace('',np.NaN)

# Create year/month/quarter features from release_date
df['Release_Price']=df['Release_Price'].str[1:].astype(float)
df['Release_Year']=df['Release_Date'].dt.year
df['Release_Quarter']=df['Release_Date'].dt.quarter
df['Release_Month']=df['Release_Date'].dt.month
df.head()
plt.figure(figsize=(13, 6))
df['Release_Month'].groupby(df['Release_Month']).count().plot(kind='bar')
plt.title('Resolution counts')
plt.xlabel('Release Month')
plt.ylabel('Count of GPU releases')
plt.show()
plt.figure(figsize=(13, 6))
df['Release_Year'].groupby(df['Release_Year']).count().plot(kind='bar')
plt.title('Count of model releases')
plt.xlabel('Release Year')
plt.ylabel('GPU model releases')
plt.show()
res=['1920 x 1080', '1600 x 900','1366 x 768','2560 x 1440','2560 x 1600', '1024 x 768', '3840 x 2160']
plt.figure(figsize=(13,6))
for i in res:
        df[df['Best_Resolution']==i]['Best_Resolution'].groupby(df['Release_Year']).count().plot(kind='line')
plt.title('Resolution counts')
plt.xlabel('Release Year')
plt.ylabel('Count of GPU releases')
plt.legend(res)
plt.show()
plt.figure(figsize=(12, 12))
df['Manufacturer'].value_counts().plot(kind='pie')
manufacturers=df['Manufacturer'].unique()
plt.figure(figsize=(13, 6))
for i in manufacturers:
      df[df['Manufacturer']==i]['Manufacturer'].groupby(df['Release_Year']).count().plot(kind='line')
plt.title('Manufacturer counts by release year')
plt.xlabel('Release Year')
plt.ylabel('Count of GPU releases')
plt.legend(manufacturers)
plt.show()
plt.figure(figsize=(13, 6))
sns.kdeplot(df[df['Release_Year']==2012]['Release_Price'])
sns.kdeplot(df[df['Release_Year']==2013]['Release_Price'])
sns.kdeplot(df[df['Release_Year']==2014]['Release_Price'])
sns.kdeplot(df[df['Release_Year']==2015]['Release_Price'])
sns.kdeplot(df[df['Release_Year']==2016]['Release_Price'])
plt.title('Price distributions by year')
plt.xlabel('Price')
plt.legend(['2012','2013','2014','2015','2016'])
plt.xlim(-100,1500)
plt.figure(figsize=(13, 6))
sns.kdeplot(df[(df['Manufacturer']=='Nvidia')&(df['Release_Price']<2000)]['Release_Price'])
#excluding expensive GPU from Nvidia, see section 5.9
sns.kdeplot(df[df['Manufacturer']=='AMD']['Release_Price'])
sns.kdeplot(df[df['Manufacturer']=='ATI']['Release_Price'])
plt.title('Price distributions by manufacturer')
plt.xlabel('Price')
plt.legend(['Nvidia','AMD','ATI'])

plt.figure(figsize=(13, 6))
df['Core_Speed'].fillna(0).astype(int).groupby(df['Release_Year']).mean().plot(kind='line')
df['Core_Speed'].fillna(0).astype(int).groupby(df['Release_Year']).max().plot(kind='line')
plt.title('Core speed in MHz by release year')
plt.xlabel('Release year')
plt.ylabel('Core speed MHz')
plt.legend(['Mean','Max'])
plt.xlim(2004,2016)
plt.show()
print(df.ix[df['Core_Speed'].fillna(0).astype(int).idxmax()][['Name','Core_Speed']])
plt.figure(figsize=(13, 6))
df['Memory'].str[:-3].fillna(0).astype(int).groupby(df['Release_Year']).mean().plot(kind='line')
df['Memory'].str[:-3].fillna(0).astype(int).groupby(df['Release_Year']).median().plot(kind='line')
plt.title('Memory in MB by release year')
plt.xlabel('Release year')
plt.ylabel('Memory MB')
plt.legend(['Mean','Median'])
plt.show()
fig, ax1=plt.subplots(figsize=(13,6))
ax = df['Memory_Bandwidth'].fillna(0).astype(float).groupby(df['Release_Year']).mean().plot(kind='line', zorder=9999); 
df['Memory_Speed'].str[:-5].fillna(0).astype(float).groupby(df['Release_Year']).mean().plot(ax=ax, kind='line',secondary_y=True)
ax.set_ylabel('Memory Speed MHz', fontsize=10);

plt.title('Mean memory bandwidth and speed by release year')
plt.xlabel('Release year')
plt.ylabel('Memory bandwidth GB/sec')
plt.show()
plt.figure(figsize=(13, 6))
df['Max_Power'].str[:-5].fillna(0).astype(float).groupby(df['Release_Year']).mean().plot(kind='line')
df['Max_Power'].str[:-5].fillna(0).astype(float).groupby(df['Release_Year']).median().plot(kind='line')
plt.title('Maximum power capacity of GPU in Watts by release year')
plt.xlabel('Release year')
plt.ylabel('Max power Watts')
plt.legend(['Mean','Median'])
plt.show()
plt.figure(figsize=(13, 6))
df['TMUs'].groupby(df['Release_Year']).mean().plot(kind='line')
df['TMUs'].groupby(df['Release_Year']).max().plot(kind='line')
plt.title('TMU value by release year')
plt.legend(['Mean','Max'])
plt.xlabel('Release year')
plt.ylabel('TMU value')
plt.xlim(2001,)
plt.show()
plt.figure(figsize=(13, 6))
df['Texture_Rate'].str[:-9].astype(float).groupby(df['Release_Year']).mean().plot(kind='line')
df['Texture_Rate'].str[:-9].astype(float).groupby(df['Release_Year']).median().plot(kind='line')
plt.title('Texture rate by release year')
plt.legend(['Mean','Median'])
plt.xlabel('Release year')
plt.ylabel('Texture rate GTexel/s')
plt.xlim(2001,)
plt.show()
plt.figure(figsize=(13, 6))
df['Pixel_Rate'].str[:-9].astype(float).groupby(df['Release_Year']).mean().plot(kind='line')
df['Pixel_Rate'].str[:-9].astype(float).groupby(df['Release_Year']).median().plot(kind='line')
df['Pixel_Rate'].str[:-9].astype(float).groupby(df['Release_Year']).max().plot(kind='line')

plt.title('Pixel rate by release year')
plt.legend(['Mean','Median','Max'])
plt.xlabel('Release year')
plt.ylabel('Texture rate')
plt.xlim(2001,)
plt.show()
plt.figure(figsize=(13, 6))
df['Process'].str[:-2].astype(float).groupby(df['Release_Year']).mean().plot(kind='line')
df['Process'].str[:-2].astype(float).groupby(df['Release_Year']).min().plot(kind='line')
plt.title('Process by release year')
plt.legend(['Mean','Min'])
plt.xlabel('Release year')
plt.ylabel('Process Nm')
plt.xlim(2001,)
plt.show()
plt.figure(figsize=(13, 6))
df['Release_Price'].groupby(df['Release_Year']).mean().plot(kind='line')
df['Release_Price'].groupby(df['Release_Year']).median().plot(kind='line')
plt.title('Price by release year')
plt.legend(['Mean','Median'])
plt.xlabel('Release year')
plt.ylabel('Price $')
plt.xlim(2006,)
plt.show()
print(df.ix[df['Release_Price'].fillna(0).astype(int).idxmax()][['Name','Release_Price','Release_Year']])
plt.figure(figsize=(13, 6))
df['Ratio_Rate']=df['Release_Price']/(df['Texture_Rate'].str[:-9].fillna(0).astype(int))
df['Ratio_Speed']=df['Release_Price']/(df['Memory_Speed'].str[:-5].fillna(0).astype(int))
df['Ratio_BW']=df['Release_Price']/(df['Memory_Bandwidth'].fillna(0).astype(int))
df['Ratio_Memory']=df['Release_Price']/(df['Memory'].str[:-3].fillna(0).astype(int))

df['Ratio_Memory'].groupby(df['Release_Year']).median().plot(kind='line')
df['Ratio_BW'].groupby(df['Release_Year']).median().plot(kind='line')
df['Ratio_Speed'].groupby(df['Release_Year']).median().plot(kind='line')
df['Ratio_Rate'].groupby(df['Release_Year']).median().plot(kind='line')
plt.title('Price/performance ratio')
plt.legend(['Texture_Rate','Memory_Speed','Memory_Bandwidth','Memory'])
plt.xlabel('Release year')
plt.ylabel('Price to metric ratio')
plt.xlim(2005,)
plt.show()
print(len(df[df.Name.str.contains('GTX 1080')]['Name']))