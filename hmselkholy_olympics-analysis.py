# import all packages and set plots to be embedded inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
color_base=sb.color_palette()[0]
%matplotlib inline


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings('ignore')

df =pd.read_csv('/kaggle/input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')
df_noc=pd.read_csv('/kaggle/input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv')
df.head()
  
df.info()
df.describe()
df_noc.info()
"""making sure that all Games column data is in Year, Season columns"""
wrong_rows=df.shape[0]-(df.Games==df.Year.astype('str')+" "+df.Season).sum()
print(wrong_rows)
df.drop(['Games','ID','Event'],axis=1,inplace=True)
df=pd.merge(df_noc,df,on='NOC',how='inner')
df.info()
df[df.region.isna()].Team.unique()

df[df.Team=='Tuvalu']

df[df.Team=='Refugee Olympic Athletes']



plt.figure(figsize=(10,5))
b=np.arange(0,df.Age.max()+1,1)
plt.hist(data=df,x='Age',color=color_base,bins=b);
t=np.arange(0,df['Age'].max()+10,10);
plt.xticks(t,t);
plt.title('Age Distribution');
plt.ylabel('Frequency');
plt.ylabel('Age');

sb.boxplot(data=df,x='Age',fliersize=1/2)
plt.title('Box Plot for Age');
plt.xlabel('Age');

'''Sports for old players'''
s=df[df.Age>65].Sport.unique()
print(s)

plt.figure(figsize=(20,5))
b=np.arange(0,df.Age.max()+5,5)
sb.countplot(data=df,x='Weight',color=color_base, );
t=np.arange(0,df['Weight'].max()+10,10);
plt.xticks(t,t);
plt.title('Weight Distribution');
plt.ylabel('Frequency');
plt.xlabel('Weight');

"""
Weight Distribution with box plot
"""
plt.figure(figsize=(20,5))

sb.boxplot(data=df,x='Weight',fliersize=2)
t=np.arange(0,df['Weight'].max()+10,10);
plt.xticks(t,t);
plt.title('Weight Distribution');

'''Sports for Fat players'''

s_w=df[df.Weight>140].Sport.unique()
print(s_w)

'Height Distribution with count plot'
plt.figure(figsize=(20,5))
b=np.arange(0,df.Age.max()+5,5)
sb.countplot(data=df,x='Height',color=color_base, );
t=np.arange(0,df['Height'].max()+10,10);
plt.xticks(t,t);
plt.title('Height Distribution');
plt.ylabel('Frequency');
plt.xlabel('Height');

plt.figure(figsize=(20,5))
sb.boxplot(data=df,x='Height',fliersize=2)
t=np.arange(df['Height'].min()-10,df['Height'].max()+5,5);
plt.xticks(t,t);
plt.title('Height Distribution');

s_h_r=df[df.Height>210].Sport.unique()
print(s_h_r)

s_h_l=df[df.Height<136].Sport.unique()
print(s_h_l)

'''plot most hosting cities'''
plt.figure(figsize=(15,10))
order=df.groupby('City').size().sort_values(ascending=False).index
ax=sb.countplot(data=df,y="City",order=order,color=(color_base))
l = ax.get_children()[0:11]
for m in l :
    m.set_color(sb.color_palette()[3])

plt.title('Most Hosting Cities');
plt.ylabel('City');
plt.ylabel('Count');

'''Did olympics became more popular over year ?
we just plot frquency over years'''
plt.figure(figsize=(20,5))
ax=sb.countplot(data=df,x="Year",color=color_base)
'''Higlight Strange Values'''

l = ax.get_children()[0:-10]
l=l[-12:-1:2]
for m in l :
    m.set_color('green');
plt.title('Did olympics became more popular over year ?\n [interesting values are in green color]',fontdict={'fontsize':15});

'''Males Vs Females '''
plt.figure(figsize=(10,5))
sb.countplot(data=df,x='Sex',)
plt.title('Males vs Females overall');

plt.figure(figsize=(12,8))
data=df.groupby('Name').size().sort_values(ascending=False).reset_index().head(10)
order=data.index
sb.barplot(data=data,y=0,x="Name",color=color_base)
plt.xticks(rotation=25);
plt.title('Most Attending Players');
plt.ylabel('Count');
'''
using dx to store frequency of players over years and season to use it in point plot to show trend line
'''
plt.figure(figsize=(25,10))
dx=df.groupby(['Season','Year']).size().reset_index()
dx['count']=dx[0]
dx.drop(0,axis=1,inplace=True)
dx.shape
'''
 show frequency of players over years and season
'''
sb.countplot(data=df,x="Year",hue='Season',hue_order=['Winter','Summer'] )
'''
show trend line
'''

sb.pointplot(data=dx,x="Year",y='count',hue='Season',hue_order=['Winter','Summer'] );


plt.figure(figsize=(25,10))
'''
trying to fill discontinuities in trend line 
'''
last=dx[dx.Year>=1992]
last_summer=last[last.Season=='Summer']
last_summer_copy=last_summer.copy()
last_summer_copy.Year=last_summer_copy.Year+2
dx=dx.append(last_summer_copy)
last_winter=last[last.Season=='Winter']
last_winter_copy=last_winter.copy()
last_winter_copy.Year=last_winter_copy.Year+2
dx=dx.append(last_winter_copy)
sb.pointplot(data=dx,x="Year",y='count',hue='Season',hue_order=['Winter','Summer'] )
ax=sb.countplot(data=df,x="Year",hue='Season',hue_order=['Winter','Summer'] )

plt.title('does winter season has fewer players?',fontdict={'fontsize':20,});
plt.ylabel('count',fontdict={'fontsize':15,});
plt.xlabel('Year',fontdict={'fontsize':15,});
plt.figure(figsize=(25,10))
sb.countplot(data=df[df.Team.isin(['Refugee Olympic Athletes', 'Tuvalu'])],x='Year',hue='Team')
plt.title('when independent teams started joining?',fontdict={'fontsize':20,});
plt.ylabel('count',fontdict={'fontsize':15,});
plt.xlabel('Year',fontdict={'fontsize':15,});
'''using data from univariate exploration '''
plt.figure(figsize=(15,10))
sb.boxplot(data=df[df.Sport.isin(s)],x='Sport',y='Age',fliersize=1/2,color=color_base)
plt.axhline(25,color='red',linestyle='--')
bbox_props = dict(boxstyle="Round,pad=0.2", fc="orange", ec="black", lw=3)
plt.annotate('Median Age=25',(3.45,25) , bbox=bbox_props)
plt.title('are there Sports that have older players as usual ?');
plt.figure(figsize=(15,10))

sb.boxplot(data=df[df.Sport.isin(s_w)],x='Sport',y='Weight',fliersize=2,color=color_base)
plt.axhline(70,color='red',linestyle='--')
bbox_props = dict(boxstyle="Round,pad=0.2", fc="orange", ec="black", lw=3)

plt.annotate('Median Weight = 70',(4.2,69),  bbox=bbox_props)
plt.title('are there Sports that have players with more weight than usual ?');
plt.figure(figsize=(15,10))
sb.boxplot(data=df[df.Sport.isin(s_h_l)],x='Sport',y='Height',fliersize=4,color=color_base)
plt.axhline(175,color='red',linestyle='--')
bbox_props = dict(boxstyle="Round,pad=0.2", fc="orange", ec="black", lw=3)
plt.annotate('Median Height = 175',(3.25,175), bbox=bbox_props);
plt.title('are there Sports that have players with less Height than usual ?');

g=sb.FacetGrid(data=df[df.Sport.isin(s)],col='Sport',col_wrap=2,height=5,aspect=2.15,sharey=False,sharex=False)
g.map(sb.countplot,'Year')
g.set_xticklabels(rotation=20)
# customize first subplot
ax = g.axes[4]  
l = ax.get_children()
l[-1].set_color((0.65,0.65,0.65,.5))

plt.figure(figsize=(20,5))
dx=df.groupby(['Sex','Year']).size().reset_index()
dx['count']=dx[0]
dx.drop(0,axis=1,inplace=True)
sb.pointplot(data=dx,x= 'Year',y='count',alpha=0.8,hue='Sex',hue_order=['M','F'] )

sb.countplot(data=df,x="Year",hue='Sex',hue_order=['M','F'] ,alpha=0.8)
plt.title('is olympics more popular for men over years ?');

df2=df.drop(df[df.Medal.isna()].index)
 
plt.figure(figsize=(10,10))
sb.regplot(data=df2,x='Age',y='Height',line_kws={'color':'red'},)
plt.title('Age vs weight');

plt.figure(figsize=(10,10))
sb.regplot(data=df2,x='Age',y='Height',line_kws={'color':'red'})
plt.title('Age vs height');
data=df.groupby(['Name','Medal']).size().sort_values(ascending=False).reset_index() 
order=data.index
data.columns=['Name', 'Medal', 'count']
vals=['Gold','Silver','Bronze']
f,ax=plt.subplots(3,1)
i=0
f.tight_layout()

for val in vals:
    sb.barplot(data=data.query('Medal==@val').head(10),y='count',x="Name",ax=ax[i],color=color_base)
    ax[i].figure.set_size_inches(20, 15)
    ax[i].set_xticklabels(ax[i].get_xticklabels(),rotation=10)
    ax[i].set_title('Top 10 players in ' +val+ ' medals')
    i=i+1
plt.subplots_adjust( hspace=0.9)
plt.figure(figsize=(20,5))

dx=df.groupby(['Season','Sex','Year']).size().reset_index()
dx['count']=dx[0]
dx.drop(0,axis=1,inplace=True)
g=sb.FacetGrid(data=dx,col='Sex',row='Season',height=5,aspect=2)
g.map(sb.barplot,"Year",'count');
g.map(sb.pointplot,"Year",'count',color='red');

df2['W_H']= df2.Weight/df2.Height 
ax = sb.heatmap(df2.drop('Year',axis=1).corr(),fmt='0.2f', linewidths=1,annot=True)


df2.dropna(inplace=True)
plt.hist2d(data = df2, x = 'Age', y = 'Weight',  
            cmap = 'viridis_r', cmin = 0.8);
plt.colorbar(label = 'Number of Players who win Medals');
plt.ylabel('weight')
plt.xlabel('Age')
plt.title('Age vs Weight');
 
plt.hist2d(data = df2, x = 'Age', y = 'Height',  
             cmap = 'viridis_r', cmin = 0.5);
plt.colorbar(label = 'Number of Players who win Medals');
plt.ylabel('Height')
plt.xlabel('Age')
plt.title('Age vs Height');
df2['W_H']=df2.Height/df2.Weight
plt.hist2d(data = df2, x = 'Age', y = 'W_H',  
             cmap = 'viridis_r', cmin = 0.5);
plt.colorbar(label = 'Number of Players who win Medals');
plt.ylabel('W_H')
plt.xlabel('Age')
plt.title('Age vs Height/weight');
