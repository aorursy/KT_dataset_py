from IPython.core.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import *



####..........Data cleaning.......###

import os
print(os.listdir("../input"))

DT=pd.read_csv("../input/data.csv",encoding = "latin1",low_memory = False)
df=pd.DataFrame(DT)



# Any results you write to the current directory are saved as output.
df['so2']=df['so2'].fillna(0).astype('str').astype('float')
df['no2']=df['no2'].fillna(0).astype('str').astype('float')
df['rspm']=df['rspm'].fillna(0).astype('str').astype('float')
df['spm']=df['spm'].fillna(0).astype('str').astype('float')
df['pm2_5']=df['pm2_5'].fillna(0).astype('str').astype('float')
df=df.rename(index=str,columns={'date':'year'})

df.drop(['stn_code','agency','location_monitoring_station'],axis=1,inplace=True)
df.info()
print('Describe evry single gas values')
df.describe()
#print('Every Single state Air pollution  Gase values')
df.groupby(['state', 'type']).count()
# defining columns of importance, which shall be used reguarly
VALUE_COLS = ['so2', 'no2', 'rspm', 'spm', 'pm2_5']

#..............State...........#

state=df.groupby(['state'],as_index=False).mean()

state

df['total']=state.sum(axis=1)
df.fillna(0.0,inplace=True)

#............. Adding a column of maximum to the 'state' dataframe...................#
state['total']=state.sum(axis=1)
print("The State with highest amount of air-pollution is :-\n\n")
print(state[state['total']==(state['total'].max())])
#............. Adding a column of minimum to the 'state'.................#
#print("The State with lowest amount of air-pollution is :-\n\n")
print(state[state['total']==(state['total'].min())])
state=state.sort_values(['total'],ascending=False)
#print("Top 5 Most Populated States are :-\n\n")
state.head()
#................ top 5  .............#

#print("Top 5 Least Populated States are :-\n\n")
state.tail().sort_values(['total'],ascending=True)
          #......state wise data...........#

    
s=df.loc[(df['state']=='Himachal Pradesh'),['state','location','so2','no2','spm','rspm','pm2_5']]
s
#.................All state Average gas data............#

df.groupby('state')[['spm','pm2_5','rspm','so2','no2']].mean()
#.....no2 gas danger zone max...#
print('show the no2 maximum value in Gujarat state')
N=df.loc[(df['state']=='Gujarat'),['no2']].max()
N
    #.....no2 gas Safe zone min...#
print('show the no2 minimum value in Gujarat state')
N1=df.loc[(df['state']=='Gujarat'),['no2']].min()
N1
     #.....no2 gas average zone average...#
print('show the no2 average value in Gujarat state')
N3=df.loc[(df['state']=='Gujarat'),['no2']].mean()
N3
    #.......Greather than and less than comparision........#
G=df.loc[(df.no2 >= 23) & (df.no2 <=25), ["state", 'location','no2']].tail(10)
G
#........which state and gass no2 pollute maximum.........#
m=df.loc[:,['no2','state','type','year']]
m.loc[m['no2'].idxmax()]
       #.....so2 gas  danger zone max...#
print('show the So2 maximum value in Chandigharh state')
S=df.loc[(df['state']=='Chandigarh'),['so2']].max()
S

#.....so2 gas safe zone min...#
print('show the So2 minimum value in Chandigharh state')
S1=df.loc[(df['state']=='Chandigarh'),['so2']].min()
S1
        #.....so2 average zone ...#
print('show the So2 Average value in Chandigharh state')
S2=df.loc[(df['state']=='Chandigarh'),['so2']].mean()
S2
  #.......Greather than and less than comparision........#

G1=df.loc[(df.so2 >= 23) & (df.so2 <=25), ["state", 'location','so2']].tail(10)
G1
#........which state and gass so2 pollute maximum.........#
m1=df.loc[:,['so2','state','type','year']]
m1.loc[m1['so2'].idxmax()]
# defining a function to find the highest ever recorded levels for a given indicator (defaults to SO2) by state
# sidenote: mostly outliers

def highest_levels_recorded(indicator="so2"):
    plt.figure(figsize=(15,5))
    ind = df[[indicator, 'location', 'state', 'year']].groupby('state', as_index=False).max()
    highest = sns.barplot(x='state', y=indicator, data=ind)
    highest.set_title("Highest ever {} levels recorded by state".format(indicator))
    plt.xticks(rotation=90)
highest_levels_recorded("so2")
#highest_levels_recorded("rspm")
#highest_levels_recorded("no2")
def highest_levels_recorded(indicator="no2"):
    plt.figure(figsize=(15,5))
    ind = df[[indicator, 'location', 'state', 'year']].groupby('state', as_index=False).max()
    highest = sns.barplot(x='state', y=indicator, data=ind)
    highest.set_title("Highest ever {} levels recorded by state".format(indicator))
    plt.xticks(rotation=90)
highest_levels_recorded("no2")
def highest_levels_recorded(indicator="rspm"):
    plt.figure(figsize=(15,5))
    ind = df[[indicator, 'location', 'state', 'year']].groupby('state', as_index=False).max()
    highest = sns.barplot(x='state', y=indicator, data=ind)
    highest.set_title("Highest ever {} levels recorded by state".format(indicator))
    plt.xticks(rotation=90)
highest_levels_recorded("rspm")
def highest_levels_recorded(indicator="spm"):
    plt.figure(figsize=(15,5))
    ind = df[[indicator, 'location', 'state', 'year']].groupby('state', as_index=False).max()
    highest = sns.barplot(x='state', y=indicator, data=ind)
    highest.set_title("Highest ever {} levels recorded by state".format(indicator))
    plt.xticks(rotation=90)
highest_levels_recorded("spm")
def highest_levels_recorded(indicator="pm2_5"):
    plt.figure(figsize=(15,5))
    ind = df[[indicator, 'location', 'state', 'year']].groupby('state', as_index=False).max()
    highest = sns.barplot(x='state', y=indicator, data=ind)
    highest.set_title("Highest ever {} levels recorded by state".format(indicator))
    plt.xticks(rotation=90)
highest_levels_recorded("pm2_5")
sns.heatmap(df.loc[:, ['state','so2', 'no2', 'rspm', 'spm', 'pm2_5']].corr(),annot=True,cmap='coolwarm')

#....................Find out the states with minmum/maximum pollution parameters...............#

fig, axes= plt.subplots(figsize=(17, 18), ncols=3)

state_wise_max_so2 = df[['state','so2']].dropna().groupby('state').median().sort_values(by='so2')

state_wise_max_no2 = df[['state','no2']].dropna().groupby('state').median().sort_values(by='no2')

state_wise_max_rspm = df[['state','rspm']].dropna().groupby('state').median().sort_values(by='rspm')

#state_wise_max_spm = df[['state','spm']].dropna().groupby('state').median().sort_values(by='spm')

#state_wise_max_pm2_5 = df[['state','pm2_5']].dropna().groupby('state').median().sort_values(by='pm2_5')


sns.barplot(x='so2', y=state_wise_max_so2.index, data=state_wise_max_so2, ax=axes[0])
axes[0].set_title("Average so2 observed in a state")

sns.barplot(x='no2', y=state_wise_max_no2.index, data=state_wise_max_no2, ax=axes[1])
axes[1].set_title("Average no2 observed in a state")

sns.barplot(x='rspm', y=state_wise_max_rspm.index, data=state_wise_max_rspm, ax=axes[2])
axes[2].set_title("Average rspm observed in a state")

#sns.barplot(x='spm', y=state_wise_max_spm.index, data=state_wise_max_spm, ax=axes[3])
#axes[3].set_title("Average spm observed in a state")

#sns.barplot(x='pm2_5', y=state_wise_max_pm2_5.index, data=state_wise_max_pm2_5, ax=axes[4])
#axes[4].set_title("Average pm2_5 observed in a state")

plt.tight_layout()
# defining a function to plot pollutant averages for a given indicator (defaults to NO2) by locations in a given state

def location_avgs(state, indicator="so2"):
    locs = df[VALUE_COLS + ['state', 'location', 'year']].groupby(['state', 'location']).mean()
    state_avgs = locs.loc[state].reset_index()
    state_avgs = state_avgs.sort_values(by=indicator,ascending=False).head(15)
    sns.set(rc={'figure.figsize':(10,10)})
    sns.barplot(x=indicator, y='location', data=state_avgs,hue='so2',dodge=False)
    plt.title("Location-wise average for {} in {}".format(indicator, state))
    #plt.xticks(rotation = 90,size=20)
location_avgs("Uttar Pradesh", "so2")
# defining a function to plot pollutant averages for a given indicator (defaults to NO2) by locations in a given state

def location_avgs(state, indicator="no2"):
    locs = df[VALUE_COLS + ['state', 'location', 'year']].groupby(['state', 'location']).mean()
    state_avgs = locs.loc[state].reset_index()
    state_avgs = state_avgs.sort_values(by=indicator,ascending=False).head(15)
    sns.set(rc={'figure.figsize':(10,10)})
    sns.barplot(x=indicator, y='location', data=state_avgs,hue='no2',dodge=False)
    plt.title("Location-wise average for {} in {}".format(indicator, state))
    #plt.xticks(rotation = 90,size=20)
location_avgs("Himachal Pradesh", "no2")
def location_avgs(state, indicator="rspm"):
    locs = df[VALUE_COLS + ['state', 'location', 'year']].groupby(['state', 'location']).mean()
    state_avgs = locs.loc[state].reset_index()
    state_avgs = state_avgs.sort_values(by=indicator,ascending=False).head(15)
    sns.set(rc={'figure.figsize':(10,10)})
    sns.barplot(x=indicator, y='location', data=state_avgs,hue='rspm',dodge=False)
    plt.title("Location-wise average for {} in {}".format(indicator, state))
    #plt.xticks(rotation = 90,size=20)
location_avgs("Maharashtra", "rspm")
def location_avgs(state, indicator="spm"):
    locs = df[VALUE_COLS + ['state', 'location', 'year']].groupby(['state', 'location']).mean()
    state_avgs = locs.loc[state].reset_index()
    state_avgs = state_avgs.sort_values(by=indicator,ascending=False).head(15)
    sns.set(rc={'figure.figsize':(10,10)})
    sns.barplot(x=indicator, y='location', data=state_avgs,hue='spm',dodge=False)
    plt.title("Location-wise average for {} in {}".format(indicator, state))
    #plt.xticks(rotation = 90,size=20)
location_avgs("Assam", "spm")
def location_avgs(state, indicator="pm2_5"):
    locs = df[VALUE_COLS + ['state', 'location', 'year']].groupby(['state', 'location']).mean()
    state_avgs = locs.loc[state].reset_index()
    state_avgs = state_avgs.sort_values(by=indicator,ascending=False).head(15)
    sns.set(rc={'figure.figsize':(10,10)})
    sns.barplot(x=indicator, y='location', data=state_avgs,hue='pm2_5',dodge=False)
    plt.title("Location-wise average for {} in {}".format(indicator, state))
    #plt.xticks(rotation = 90,size=20)
location_avgs("Gujarat", "pm2_5")
labels='no2','s02'
sizes=[876.0,909.0]
colors=['gold','green']
explode=(0,0)

plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True,startangle=300)
plt.legend(labels,loc='best')
plt.axis('equal')
plt.show()
labels='rspm','spm','pm2_5'
sizes=[307.0,380.0,504.0]
colors=['gold','green','blue']
explode=(0,0,0)

plt.pie(sizes,labels=labels, colors=colors,radius=1,autopct='%2.f%%', shadow=True, startangle=90)
plt.legend( labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()
labels='rspm','spm','pm2_5','no2','s02'
sizes=[6307.0,3380.0,504.0,876.0,909.0]
colors=['skyblue','green','blue','yellow','red']
explode=(0.1,0.3,0.5,0.4,0)

plt.pie(sizes,explode=explode,labels=labels,colors=colors,radius=4,autopct='%1.1f%%',shadow=True,startangle=300)
plt.axis('equal')
plt.title('          Bihar',size=20)

plt.show()
d=df.groupby(['state'])['no2'].sum().sort_values(kind='mergesort',ascending=False).head(10)
d.plot(kind='pie',autopct='%.0f%%',radius=2,subplots=True)