import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/daily-temperature-of-major-cities/city_temperature.csv')

df.head()
df.isna().sum()
df.describe()
df['Year'].unique()
df = df.drop('State',axis=1)

df.loc[df['Year']==200,'Year']=2000

df.loc[df['Year']==201,'Year']=2010

df.head()
s= df.groupby(['Region'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)

s.style.background_gradient(cmap='Purples')



plt.figure(figsize=(18,8))

sns.barplot(x='Region', y= 'AvgTemperature',data=s,palette='hsv_r')

plt.title('AVERAGE MEAN TEMPERATURE OF DIFFERENT REGIONS')

plt.show()

a= df.groupby(['Year','Region'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)

a.head(20).style.background_gradient(cmap='Blues')

plt.figure(figsize=(15,8))

sns.lineplot(x='Year',y='AvgTemperature',hue='Region',data=a,palette='hsv')

plt.grid()

plt.title('YEAR-WISE AVERAGE MEAN TEMPERATURE OF DIFFERENT REGIONS')

plt.show()
b= df.groupby(['Region','Month'])['AvgTemperature'].max().reset_index().sort_values(by='AvgTemperature',ascending=False)

b.head(20).style.background_gradient(cmap='Oranges')

plt.figure(figsize=(15,8))

sns.barplot(x='Month', y= 'AvgTemperature',data=b,hue='Region',palette='hsv',saturation=.80)

plt.title('VARIATION OF MAXIMUM TEMPERATURE OVER THE MONTHS')

plt.show()
c= df.groupby(['Region','Year'])['AvgTemperature'].max().reset_index().sort_values(by='AvgTemperature',ascending=False)

c.head(20).style.background_gradient(cmap='Greens')

plt.figure(figsize=(15,8))

sns.scatterplot(x='Year',y='AvgTemperature',data=c,hue='Region',palette='hsv_r',style='Region')

plt.title(' VARIATION OF MAXIMUM TEMPERATURE OVER THE YEARS')

plt.show()

c= df.groupby(['Country','City'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False).head(20)

c.style.background_gradient(cmap='Reds')

plt.figure(figsize=(8,10))

sns.barplot(x='AvgTemperature',y='City',data=c,palette='hsv_r')

plt.title('VARIATION OF MEAN TEMPERATURE FOR TOP 20 COUNTRIES')

plt.show()
africa=df[df['Region']=='Africa']

d= africa.groupby(['Month'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)





asia=df[df['Region']=='Asia']

e= asia.groupby(['Month'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)





mid_est=df[df['Region']=='Middle East']

p= mid_est.groupby(['Month'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)





n_amer=df[df['Region']=='North America']

q= n_amer.groupby(['Month'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)





eup=df[df['Region']=='Europe']

r= eup.groupby(['Month'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)





sth=df[df['Region']=='South/Central America & Carribean']

s= sth.groupby(['Month'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)





aus=df[df['Region']=='Australia/South Pacific']

m= aus.groupby(['Month'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)







plt.figure(figsize=(15,20))

plt.subplot(4,2,1)

sns.barplot(x='Month',y='AvgTemperature',data=d,palette='hsv')

plt.title('Africa')



plt.subplot(4,2,2,)

sns.barplot(x='Month',y='AvgTemperature',data=e,palette='hsv')

plt.title('Asia')





plt.subplot(4,2,3)

sns.barplot(x='Month',y='AvgTemperature',data=p,palette='hsv')

plt.title('Middle East')





plt.subplot(4,2,4)

sns.barplot(x='Month',y='AvgTemperature',data=q,palette='hsv')

plt.title('North America')





plt.subplot(4,2,5)

sns.barplot(x='Month',y='AvgTemperature',data=r,palette='hsv')

plt.title('Europe')





plt.subplot(4,2,6)

sns.barplot(x='Month',y='AvgTemperature',data=s,palette='hsv')

plt.title('South/Central America & Carribean')





plt.subplot(4,2,7)

sns.barplot(x='Month',y='AvgTemperature',data=m,palette='hsv')

plt.title('Australia/South Pacific')



plt.show()

ind=df[df['Country']=='India']

x= ind.groupby(['Year'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)

x.style.background_gradient(cmap='hsv')

plt.figure(figsize=(15,8))

sns.lineplot(x='Year',y='AvgTemperature',data=x,color='r')

plt.grid()

plt.title('Mean Temp. Variation in India') 

plt.show()
ind=df[df['Country']=='India']

x= ind.groupby(['City','Year'])['AvgTemperature'].max().reset_index().sort_values(by='AvgTemperature',ascending=False)

x.head(20).style.background_gradient(cmap='Blues')

plt.figure(figsize=(15,8))

sns.lineplot(x='Year',y='AvgTemperature',data=x,hue='City',style='City',markers=['o','*','^','>'])

plt.grid()

plt.title('Mean Temp. Variation Of Cities of India')

plt.show()


mask1=df['Country']=='India'

mask2=df['City']=='Delhi'



ind=df[mask1 & mask2 ]





y= ind.groupby(['Year','City','Month'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)

y.head(20).style.background_gradient(cmap='PiYG')

plt.figure(figsize=(15,12))

plt.subplot(2,1,1)

sns.barplot(x='Year',y='AvgTemperature',data=y,palette='hsv_r')

plt.title('Mean Temp. Variation Of Delhi(Yearly)')



plt.subplot(2,1,2)

sns.barplot(x='Month',y='AvgTemperature',data=y,palette='hsv')

plt.title('Mean Temp. Variation Of Delhi(Monthly)')



plt.show()



mask1=df['Country']=='India'

mask2=df['Year']==2020



ind=df[mask1 & mask2 ]





k= ind.groupby(['Year','City','Month'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)

k.style.background_gradient(cmap='Greens')

plt.figure(figsize=(15,8))

sns.lineplot(x='Month',y='AvgTemperature',data=k,hue='City',style='City',markers=['*','o','<','>'])

plt.grid()

plt.title('Mean Temp. Variation Of Cities of India in 2020')

plt.show()