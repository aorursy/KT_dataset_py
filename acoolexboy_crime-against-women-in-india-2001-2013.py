import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
df1=pd.read_csv("../input/data1.csv")
df2=pd.read_csv("../input/data2.csv")
df=pd.concat([df1,df2])
df
df=df.rename(columns={'STATE/UT':'STATE','Kidnapping and Abduction':'Kidnapping','Assault on women with intent to outrage her modesty':'Assault','Insult to modesty of Women':'Insult','Cruelty by Husband or his Relatives':'Cruelty','Importation of Girls':'Importation','Dowry Deaths':'Dowry'})
df
df1=pd.read_csv("../input/data1.csv")
df2=pd.read_csv("../input/data2.csv")
df=pd.concat([df1,df2])
df
df=df.rename(columns={'STATE/UT':'STATE','Kidnapping and Abduction':'Kidnapping','Assault on women with intent to outrage her modesty':'Assault','Insult to modesty of Women':'Insult','Cruelty by Husband or his Relatives':'Cruelty','Importation of Girls':'Importation','Dowry Deaths':'Dowry'})
df
states = df.STATE.unique()
print(states)
s=df.set_index('Year')
df1 = df[df.DISTRICT != 'ZZ TOTAL']
df2=df1[df1.DISTRICT != 'TOTAL']
df2.head(50)
df3=df.groupby('Year').sum()
df3

df3.describe()

df2=df.groupby(['Year','STATE']).sum()
df2

df_2012=df[df['Year']==2012]
plt.figure(figsize=(25,25))
x=df_2012['STATE']
y=df_2012['Rape']
plt.barh(x,y)
plt.show()

df_2012=df[df['Year']==2012]
plt.figure(figsize=(25,25))
x=df_2012['STATE']
y=df_2012['Dowry']
plt.barh(x,y)
plt.show()
fig, axarr = plt.subplots(2, 2, figsize=(25, 100))

df_2010['Assault'].value_counts().sort_index().plot.barh(
    ax=axarr[0][0]
)

df_2011['Assault'].value_counts().sort_index().plot.barh(
    ax=axarr[0][1])

df_2012['Assault'].value_counts().sort_index().plot.barh(
    ax=axarr[1][0]
)

df_2013['Assault'].value_counts().sort_index().plot.barh(
    ax=axarr[1][1])
crimes_total = df[df['DISTRICT'] == 'TOTAL']
crimes_total
crimes_total_2001 = crimes_total[crimes_total['Year'] == 2001]

x = crimes_total_2001['STATE'].values
y = crimes_total_2001['Kidnapping'].values

fig, ax = plt.subplots()
crime_Kidnapping= crimes_total_2001['STATE'].values
y_pos = np.arange(len(crime_Kidnapping))
performance = crimes_total_2001['Kidnapping'].values
ax.barh(y_pos, performance, align='center',color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(crime_Kidnapping)
ax.invert_yaxis()  
ax.set_xlabel('Kidnappings')
ax.set_title('Kidnapping VS STATE')
fig.set_size_inches(20, 18, forward=True)
plt.show()

crimes_total_2012 = crimes_total[crimes_total['Year'] == 2012]



x = crimes_total_2012['STATE'].values
y = crimes_total_2012['Assault'].values


fig, ax = plt.subplots()
crime_Assault = crimes_total_2012['STATE'].values
y_pos = np.arange(len(crime_Assault))
performance = crimes_total_2012['Assault'].values
ax.barh(y_pos, performance, align='center',color='red', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(crime_Assault)
ax.invert_yaxis()  
ax.set_xlabel('Assault')
ax.set_title('Assault VS STATE')
fig.set_size_inches(20, 18, forward=True)
plt.show()
plt.figure(figsize=(25,25))
x=crimes_total['Year']
y=crimes_total['Dowry']
plt.barh(x,y)
plt.xlabel('No of Dowry Deaths',  fontsize=20)
plt.ylabel('Year',  fontsize=20)
plt.title('Yearwise Dowry Deaths',fontsize=20)
plt.show()
