import pandas as pd
import matplotlib.pyplot as plt
import folium
covid = pd.read_html('https://www.mohfw.gov.in/')
df = covid[0]
df
df = df.rename(columns={'Name of State / UT':'State/UT',
                   'Total Confirmed cases (Including 111 foreign Nationals)':'Total',
                   'Cured/Discharged/Migrated':'Cured'})
df = df.drop('S. No.',axis=1)
df.set_index('State/UT',inplace=True)
df = df[:-4]
df = df[['Cured','Total','Death']].astype(int)
df['Active'] = df['Total']-(df['Cured']+df['Death'])
df = df[['Active','Cured','Death','Total']]
df.shape
print('Dimensions of Data Frame',df.shape)
df.head()
df.tail()
total = df.groupby('State/UT')['Total'].sum()
total = total.sort_values(ascending=False).to_frame()
total
total.plot(kind='bar',figsize=(11,5),color='#F95700FF')
plt.title('Affected states by COVID-19',fontsize=20)
plt.ylabel('Victims Affected',fontsize=18)
plt.xlabel('State/UT',fontsize=17)
plt.show()
tdf = total.head()
tdf
tdf.plot(kind='area',figsize=(11,5),color='#00B1D2FF')
plt.title('Top 5  states affected by COVID-19',fontsize=20)
plt.xlabel('State/UT',fontsize=17)
plt.ylabel('Victims Affected',fontsize=18)
plt.show()
ldf = total.tail()
ldf
ldf.plot(kind='line',figsize=(11,5),
           color='#12C0F7',
           marker='o',
           markersize=12,
           markerfacecolor='#F73A68',
           linewidth=3)
plt.title('Least affected states by COVID-19',fontsize=20)
plt.xlabel('State/UT',fontsize=17)
plt.ylabel('Victims Affected',fontsize=18)
plt.show()
active = df.groupby('State/UT')['Active'].sum()
active.plot(kind='bar',figsize=(11,5),color='#0547F8')
plt.title('Active cases in India ',fontsize=20)
plt.xlabel('State/UT',fontsize=17)
plt.ylabel('Victims Affected',fontsize=18)
plt.show()
active = active.sort_values(ascending=False).to_frame()
adf = active.head()
aldf = active.tail()
adf.plot(kind='line',figsize=(11,5),
           color='#00539CFF',
           marker='o',
           markersize=11,
           markerfacecolor='#FFD662FF',
           linewidth=3)
plt.title('Top 5 states are in Active',fontsize=20)
plt.xlabel('State/UT',fontsize=17)
plt.ylabel('Victims Affected',fontsize=18)
plt.show()
aldf.plot(kind='line',figsize=(11,5),
           color='#2BAE66FF',
           marker='o',
           markersize=12,
           markerfacecolor='#FCF6F5FF',
           linewidth=3)
plt.title('Have less impact by Covid-19',fontsize=20)
plt.xlabel('State/UT',fontsize=17)
plt.ylabel('Victims Affected',fontsize=18)
plt.show()
recovery = df.groupby('State/UT')['Cured'].sum()
recovery.plot(kind='bar',figsize=(11,5),color='#F93822FF')
plt.title('Cured patients from Covid-19',fontsize=20)
plt.xlabel('State/UT',fontsize=17)
plt.ylabel('Victims Affected',fontsize=18)
plt.show()
recovery = recovery.sort_values(ascending=False).to_frame()
fdf = recovery.head()
fdf
fdf.plot(kind='line',figsize=(11,5),
           color='#00A4CCFF',
           marker='o',
           markersize=12,
           markerfacecolor='#F95700FF',
           linewidth=3)
plt.title('Fastest reovering States',fontsize=20)
plt.xlabel('State/UT',fontsize=17)
plt.ylabel('Victims Affected',fontsize=18)
plt.show()
death = df.groupby('State/UT')['Death'].sum()
death = death.sort_values(ascending=False).to_frame()
tdf.plot(kind='area',figsize=(11,5),color='#00A4CCFF')
plt.title('Deaths by Covid-19',fontsize=20)
plt.xlabel('State/UT',fontsize=17)
plt.ylabel('Victims Affected',fontsize=18)
plt.show()
tddf = death.head()
tddf

tddf.plot(kind='line',figsize=(11,5),
           color='#606060FF',
           marker='o',
           markersize=12,
           markerfacecolor='#D6ED17FF',
           linewidth=3)
plt.title('High  Death Impact States',fontsize=20)
plt.xlabel('State/UT',fontsize=17)
plt.ylabel('Victims Affected',fontsize=18)
plt.show()
lddf = death.tail()
lddf.plot(kind='line',figsize=(11,5),
           color='#101820FF',
           marker='o',
           markersize=12,
           markerfacecolor='#FEE715FF',
           linewidth=3)
plt.title('No Death Impact States',fontsize=20)
plt.xlabel('State/UT',fontsize=17)
plt.ylabel('Victims Affected',fontsize=18)
plt.show()