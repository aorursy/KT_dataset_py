import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

import random

from wordcloud import WordCloud, STOPWORDS

pd.set_option('display.max_column',None)
pd.set_option('display.max_rows',None)
Guns=pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")
Guns.head()
Guns.shape[0]
Guns.describe()#lots of null values everywhere
per=[]
for i in Guns.columns:
    num=Guns[i].isnull().sum()
    final=(num/Guns.shape[0])*100
    per.append(final)

d={'Col': Guns.columns,'%null': per}
nulls=pd.DataFrame(data=d)
nulls
plt.figure(figsize=(18,12))
state=Guns['state'].value_counts()
sns.barplot(state.values,state.index)
plt.xlabel("Number of incidences",fontsize=15)
plt.ylabel("States",fontsize=15)
plt.title("Recoreded incidences in states",fontsize=20)
sns.despine(left=True,right=True)
plt.show()
plt.figure(figsize=(18,12))
state=Guns['city_or_county'].value_counts()[:20]
sns.barplot(state.values,state.index)
plt.xlabel("Number of incidences",fontsize=15)
plt.ylabel("cities",fontsize=15)
plt.title("Recored incidences in cities",fontsize=20)
sns.despine(left=True,right=True)
plt.show()
Guns['date']=pd.to_datetime(Guns['date'],format='%Y-%m-%d')
Guns['date'].head()
Guns['year']=Guns['date'].dt.year
state_lst=[Guns['state'].unique()]
year_lst=[Guns['date'].dt.year.unique()]

state_lst_new=[]
for i in range(0,51):
    new=state_lst[0][i]
    state_lst_new.append(new)
    
year_lst_new=[]
for i in range(0,6):
    new=year_lst[0][i]
    year_lst_new.append(new)

plt.figure(figsize=(18,9))
for state in state_lst_new:
    yearly_incd=[]
    for year in year_lst_new:
        my= Guns.loc[Guns['state']==state]
        sum=my.loc[my['year']==year]
        sol=sum.shape[0]
        yearly_incd.append(sol)
    plt.plot(yearly_incd,label=state)
plt.xticks(np.arange(6),tuple(year_lst_new),rotation=60)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
plt.show()
years_killed=Guns.groupby(Guns['year']).sum()
x=years_killed['n_killed'].index.tolist()
y=years_killed['n_killed'].values.tolist()
z=years_killed['n_injured'].values.tolist()

#create style trace
trace0=go.Scatter(
x = x,
y = y,
name='no. of people killed',
  line = dict(
        color = ('rgb(205, 12, 24)'),
        width = 4,dash='dot')
)
trace1=go.Scatter(
x = x,
y = z,
name='no. of people injured',
  line = dict(
        color = ('rgb(10, 205, 26)'),
        width = 4,dash='dot')
)
trace2=go.Scatter(
x = x,
y = [y+z for y,z in zip(y,z)],
name='Total no. of people effected',
  line = dict(
        color = ('rgb(20, 20, 205)'),
        width = 4,dash='dot')
)


data=[trace0,trace1,trace2]

#edit layout

layout=dict(title='people killed or injured every year',
           xaxis=dict(title='Years'),
           yaxis=dict(title='NO. of people killed or injured'))

fig = dict(data=data, layout=layout)
py.iplot(fig , filename='styled-line')
state_killed=Guns.groupby(Guns['state']).sum()
sk_x=state_killed['n_killed'].index.tolist()
sk_y=state_killed['n_killed'].values.tolist()
si=state_killed['n_injured'].values.tolist()

trace1=go.Scatter(
x=sk_x,
y=sk_y,
name='people killed')

trace2=go.Scatter(
x=sk_x,
y=si,
name='people injured',
yaxis='y2')
data=[trace1,trace2]

layout = go.Layout(
    title='Incidences in states',
    xaxis=dict(title='states'),
    yaxis=dict(
        title='People killed',
        titlefont=dict(
            color='rgb(140,38,78)')
    ),
    yaxis2=dict(
        title='people injured',
        titlefont=dict(
            color='rgb(148, 103, 189)'
        ),
        tickfont=dict(
            color='rgb(148, 103, 189)'
        ),
        overlaying='y',
        side='right'
    )
 )
                 
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='multiple-axes-double.html')
            
cd=Guns[np.isfinite(Guns['congressional_district'])]
my_cd=cd['congressional_district'].value_counts()

plt.figure(figsize=(28,12))
sns.barplot(my_cd.index,my_cd.values)
plt.xlabel("Congressional district",fontsize=12)
plt.ylabel("NO. of incidences",fontsize=12)
plt.title("Gun abuse in different CD ",fontsize=16)
plt.show()
type=Guns.dropna(how='any',axis=0)
my_type=type['gun_type'].values.tolist()
del( my_type[5:11])

my_set=set()
for guns in my_type:
    if len(guns)<=18:
        adds=guns.split("::")[1]
        my_set.add(adds)
    else:
        my_item=[]
        my_items=[]
        lst1=guns.split("||")
        for item in lst1:
            my=item.split("::")
            my_item.append(my)
        for items in my_item:
            adds=items[1]
            my_items.append(adds)
        for adding in my_items:
            my_set.add(adding)
        
        
remove=['45 Auto||1','9mm||1','Handgun||1','Rifle||1']
for rem in remove:
    my_set.remove(rem)
my_set

str_set=[]
for e in my_set:
    string=str(e)
    str_set.append(string)
str_set


stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color='black',
                          stopwords=stopwords,
                         ).generate(' '.join(str_set))
print(wordcloud)
fig = plt.figure(figsize=(14,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
item=Guns['state'].value_counts().index.tolist()
item_size=Guns['state'].value_counts().values.tolist()

cities = []
scale = 250


for i in range(len(item)):
    lim = item[i]
    df_sub = Guns.loc[Guns['state']==lim][:1]
    city = dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = df_sub['longitude'],
        lat = df_sub['latitude'],
        text = item[i] + '<br>Gun abuse ' + str(item_size[i]),
        marker = dict(
            size = item_size[i]/scale,
            line = dict(width=0.5, color='rgb(40,40,40)'),
            sizemode = 'area'
        ),
        name = lim )
    cities.append(city)

layout = dict(
        title = 'Gun abuse around USA state',
        showlegend = True,
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            landcolor = 'rgb(217, 217, 217)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        ),
    )

fig = dict( data=cities, layout=layout )
py.iplot( fig, validate=False, filename='d3-bubble-map-populations' )
plt.savefig('abc.png')    
item=Guns['city_or_county'].value_counts()[:1000].index.tolist()
item_size=Guns['city_or_county'].value_counts()[:5000].values.tolist()

cities = []



for i in range(len(item)):
    lim = item[i]
    df_sub = Guns.loc[Guns['city_or_county']==lim][:1]
    city = dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = df_sub['longitude'],
        lat = df_sub['latitude'],
        text = df_sub['state'],
        marker = dict(
            line = dict(width=0.5, color='rgb(40,40,40)'),
            sizemode = 'area'
        ),
        name = lim )
    cities.append(city)

layout = dict(
        title = 'Gun abbuse around USA',
        showlegend = True,
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            landcolor = 'rgb(217, 217, 217)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        ),
    )

fig = dict( data=cities, layout=layout )
py.iplot( fig, validate=False, filename='d3-bubble-map-populations-b' )
Guns_s=Guns.dropna(subset=['participant_age'])

all_age=[]
guns_age=Guns_s['participant_age'].tolist()
for age in guns_age:
    if len(age)>5:
        x=age.split("||")
        for fin_age in x:
            y=fin_age.split("::")[0]
            all_age.append(y)
    else:
        k=age.split("::")[0]
        all_age.append(k)
len(all_age)
count=0
mount=0
all_age_f=[]
for aa in all_age:
    if len(aa)>1:
        count = count+1
    else:
        mount = mount+1
        all_age_f.append(int(aa))
group_labels = ['distplot']
rand_data=random.sample(all_age_f,50000)
hist_data=[rand_data]

fig = ff.create_distplot(hist_data, group_labels)
py.iplot(fig, filename='Basic Distplot')
Guns['singles']=1
Guns['date']=pd.to_datetime(Guns['date'],format='%Y-%m-%d')
time_s=Guns.groupby(['date']).sum()
trace1 = go.Scatter(
                x=time_s.index,
                y=time_s['n_killed'],
                name = "People killed",
                line = dict(color = '#17BECF'),
                opacity = 0.8)

trace2 = go.Scatter(
                x=time_s.index,
                y=time_s['n_injured'],
                name = "Injured",
                line = dict(color = '#3F3B3C'),
                opacity = 0.8)
trace3 = go.Scatter(
                x=time_s.index,
                y=time_s['singles'],
                name = "Incidences",
                line = dict(color = '#3A6A3A'),
                opacity = 0.8)
data = [trace1,trace2,trace3]

layout = dict(
    title = "Over time stats",
    xaxis = dict(
        range = ['2013-01-01','2018-04-01'])
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename = "Manually Set Range")
plt.savefig('xyz.png')  
