# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

np.random.seed(sum(map(ord, 'calmap')))

# linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
coordinate=pd.read_csv("/kaggle/input/covid19/world_coordinate.csv")

coordinate
coordinate.rename(columns={

    

    'Province/State':'province',

    'Country/Region':'country',

    'Lat':'lat',

    'Long':'long',

    

},inplace=True)

coordinate
submission=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')

submission.head(),

submission.shape
train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

train
train.rename(columns={

    'Id':'id',

    'Province_State':'province',

    'Country_Region':'country',

    'Date':'date',

    'ConfirmedCases':'confirmed',

    'Fatalities':'deaths'

},inplace=True)

train
test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

test
train_1=train.merge(coordinate,on='country')

train_1
test.rename(columns={

    'ForecastId':'forecastid',

    'Province_State':'province',

    'Country_Region':'country',

    'Date':'date'

},inplace=True)
train['province'].unique()
test['province'].unique()
train.info()
train['province'].isna().sum()
train['country'].unique()
train['country'].value_counts()
!pip install --upgrade calmap
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors



from IPython.display import display

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.preprocessing import PolynomialFeatures

import datetime

import time

import warnings

warnings.filterwarnings('ignore')

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



import folium

from datetime import timedelta

# color pallette

cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801' 

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters() 

%matplotlib inline

from plotly.offline import plot, iplot, init_notebook_mode

init_notebook_mode(connected=True)
cols=train.keys()

cols
confirmed=train.loc[:,cols[4]:cols[-2]]

confirmed





deaths=train.loc[:,cols[5]:cols[-1]]

deaths
temp=train.groupby('date')['confirmed','deaths'].sum().reset_index()

temp=temp[temp['date']==max(temp['date'])].reset_index(drop=True)

tm=temp.melt(id_vars="date",value_vars=['confirmed','deaths'])

fig=px.treemap(tm,path=["variable"],values="value",height=225,width=1200,color_discrete_sequence=[act,rec])

fig.data[0].textinfo='label+text+value'

fig.show()
temp=train.groupby('date')['confirmed','deaths'].sum().reset_index()

temp=temp.melt(id_vars="date",value_vars=["deaths","confirmed"],var_name='Case',value_name='Count')

temp.head()

fig=px.area(temp,x='date',y='Count',color='Case',height=600,title='Cases over time',color_discrete_sequence=[rec,dth])

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
full_grouped=train.groupby(['date','country',])['confirmed','deaths'].sum().reset_index()

#Day wise

day_wise=full_grouped.groupby('date')['confirmed','deaths'].sum().reset_index()

#number of cases per 100 cases

day_wise['deaths/100 Cases']=round((day_wise['deaths']/day_wise['confirmed'])*100,2)

#no of countries

day_wise['No.of countries']=full_grouped[full_grouped['confirmed']!=0].groupby('date')['country'].unique().apply(len).values

cols=['deaths/100 Cases']

day_wise[cols]=day_wise[cols].fillna(0)
#Country wise

country_wise=full_grouped[full_grouped['date']==max(full_grouped['date'])].reset_index(drop=True).drop('date',axis=1)

#group by country

country_wise=country_wise.groupby('country')['confirmed','deaths'].sum().reset_index()

#per 100 cases

country_wise['deaths/100 Cases']=round((country_wise['deaths']/country_wise['confirmed'])*100,2)

cols=['deaths/100 Cases']

country_wise[cols]=country_wise[cols].fillna(0)

country_wise.head()
temp_1=full_grouped.sort_values(by='deaths',ascending=False)

temp_1=temp_1.reset_index(drop=True)

temp_1.style.background_gradient(cmap='Blues')
#Over the time

fig=px.choropleth(full_grouped,locations='country',locationmode='country names',color=np.log(full_grouped['confirmed']),hover_name='country',animation_frame=full_grouped['date'],title='Cases over time',color_continuous_scale=px.colors.sequential.Magenta)

fig.update(layout_coloraxis_showscale=False)

fig.show()
#Confirmed

fig_c=px.choropleth(country_wise,locations="country",locationmode='country names',color=np.log(country_wise["confirmed"]),hover_name="country",hover_data=['confirmed'])

#Deaths

temp=country_wise[country_wise['deaths']>0]

fig_d=px.choropleth(temp,locations='country',locationmode='country names',color=np.log(temp['deaths']),hover_name='country',hover_data=['deaths'])

#plot

fig=make_subplots(rows=1,cols=2,subplot_titles=['confirmed','deaths'],specs=[[{"type":"choropleth"},{"type":"choropleth"}]])

fig.add_trace(fig_c['data'][0],row=1,col=1)

fig.add_trace(fig_d['data'][0],row=1,col=2)

fig.update(layout_coloraxis_showscale=False)

fig.show()

fig_c=px.bar(day_wise,x="date",y="confirmed",color_discrete_sequence=[act])

fig_d=px.bar(day_wise,x="date",y="deaths",color_discrete_sequence=[dth])

fig=make_subplots(rows=1,cols=2,shared_xaxes=False,horizontal_spacing=0.1,subplot_titles=('Confirmed case','Deaths reported'))

fig.add_trace(fig_c['data'][0],row=1,col=1)

fig.add_trace(fig_d['data'][0],row=1,col=2)

fig.update_layout(height=460)

fig.show()

#-----------------

fig_1=px.line(day_wise,x='date',y='deaths/100 Cases',color_discrete_sequence=[dth])

fig=make_subplots(rows=1,cols=1,shared_xaxes=False,subplot_titles=('Deaths/100 Cases'))

fig.add_trace(fig_1['data'][0],row=1,col=1)

fig.update_layout(height=460)

fig=px.scatter(country_wise.sort_values('deaths',ascending=False).iloc[:15,:],x="confirmed",y='deaths',color='country',size='confirmed',height=700,text='country',log_x=True,log_y=True,title="Deaths Vs Confirmed (Scale is in log10)")

fig.update_traces(textposition='top center')

fig.update_layout(showlegend=False)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
fig=px.bar(full_grouped,x='date',y='confirmed',color='country',height=600,title='Confirmed',color_discrete_sequence=px.colors.cyclical.mygbm)

fig.show()

fig=px.bar(full_grouped,x='date',y='deaths',color='country',height=600,title='Deaths',color_discrete_sequence=px.colors.cyclical.mygbm)

fig.show()
fig=px.line(full_grouped,x='date',y='confirmed',color='country',height=600,title='Confirmed',color_discrete_sequence=px.colors.cyclical.mygbm)

fig.show()

fig=px.line(full_grouped,x='date',y='deaths',color='country',height=600,title='Deaths',color_discrete_sequence=px.colors.cyclical.mygbm)

fig.show()
full_latest=train[train['date']==max(train['date'])]

fig=px.treemap(full_latest.sort_values(by='confirmed',ascending=False).reset_index(drop=True),path=["country"],values='confirmed',height=700,title="Number of Confirmed Cases",color_discrete_sequence=px.colors.qualitative.Dark2)

fig.data[0].textinfo='label+text+value'

fig.show()



fig=px.treemap(full_latest.sort_values(by='deaths',ascending=False).reset_index(drop=True),

              path=["country"],values='deaths',height=700,title="Number of Deaths Cases",color_discrete_sequence=px.colors.qualitative.Dark2)

fig.data[0].textinfo='label+text+value'

fig.show()

from IPython.core.display import HTML

HTML('''<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1571387"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')
import math

temp=train.groupby(['country','date',])['confirmed','deaths']

temp=temp.sum().diff().reset_index()

mask=temp['country']!=temp['country'].shift(1)

temp.loc[mask,'confirmed']=np.nan

temp.loc[mask,'deaths']=np.nan



countries=temp['country'].unique()

n_cols=4

n_rows=math.ceil(len(countries)/n_cols)

fig=make_subplots(rows=n_rows,cols=n_cols,shared_xaxes=False,subplot_titles=countries)

for ind,country in enumerate(countries):

    row=int((ind/n_cols)+1)

    col=int((ind%n_cols)+1)

    fig.add_trace(go.Bar(x=temp['date'],y=temp.loc[temp['country']==country,'confirmed'],name=country),row=row,col=col)

fig.update_layout(height=8000,title_text="No.of new cases in each country")

fig.show()

import seaborn as sns

f,ax=plt.subplots(figsize=(120,40))

data=train[['country','confirmed','deaths']]

data.sort_values('confirmed',ascending=False,inplace=True)

sns.set_color_codes("pastel")

sns.barplot(x="confirmed",y="country",data=train,label="Confirmed",color="r")

sns.set_color_codes("muted")

sns.barplot(x='deaths',y='country',data=train,label="Deaths",color="g")

ax.legend(ncol=2,loc="lower right",frameon=True)

ax.set(xlim=(0,135),ylabel="",xlabel="cases")

sns.despine(left=True,bottom=True)
#confirmed-deaths

fig_c=px.bar(country_wise.sort_values('confirmed').tail(25),x='confirmed',y='country',orientation='h',color_discrete_sequence=[act])

fig_d=px.bar(country_wise.sort_values('deaths').tail(25),x='deaths',y='country',orientation='h',color_discrete_sequence=[dth])





#death-recoverd /100 cases

fig_dc=px.bar(country_wise.sort_values('deaths/100 Cases').tail(25),x="deaths/100 Cases",y='country',text='deaths/100 Cases',orientation='h',color_discrete_sequence=['#f38181'])

#plot

fig=make_subplots(rows=3,cols=2,shared_xaxes=False,horizontal_spacing=0.14,vertical_spacing=0.08,subplot_titles=('Confirmed Cases','Deaths reported','Deaths/100 cases'))

fig.add_trace(fig_c['data'][0],row=1,col=1)

fig.add_trace(fig_d['data'][0],row=1,col=2)

fig.add_trace(fig_dc['data'][0],row=2,col=1)

fig.update_layout(height=30000)
total_confirmed_China=train[train['country']=='China'].groupby(['date']).agg({'confirmed':['sum']})

total_deaths_China=train[train['country']=='China'].groupby(['date']).agg({'deaths':['sum']})

total_China=total_confirmed_China.join(total_deaths_China)

total_confirmed_Italy=train[train['country']=='Italy'].groupby(['date']).agg({'confirmed':['sum']})

total_deaths_Italy=train[train['country']=='Italy'].groupby(['date']).agg({'deaths':['sum']})

total_Italy=total_confirmed_Italy.join(total_deaths_Italy)

total_confirmed_Spain=train[train['country']=='Spain'].groupby(['date']).agg({'confirmed':['sum']})

total_deaths_Spain=train[train['country']=='Spain'].groupby(['date']).agg({'deaths':['sum']})

total_Spain=total_confirmed_Spain.join(total_deaths_Spain)

total_confirmed_US=train[train['country']=='US'].groupby(['date']).agg({'confirmed':['sum']})

total_deaths_US=train[train['country']=='US'].groupby(['date']).agg({'deaths':['sum']})

total_US=total_confirmed_US.join(total_deaths_US)

total_confirmed_India=train[train['country']=='India'].groupby(['date']).agg({'confirmed':['sum']})

total_deaths_India=train[train['country']=='India'].groupby(['date']).agg({'deaths':['sum']})

total_India=total_confirmed_India.join(total_deaths_India)

plt.figure(figsize=(17,10))

plt.subplot(3,2,1)

total_China.plot(ax=plt.gca(),title='China')

plt.ylabel('Confirmed infection cases')

plt.subplot(3,2,2)

total_Italy.plot(ax=plt.gca(),title='Italy')

plt.ylabel('Confirmed infection cases')

plt.subplot(3,2,3)

total_Spain.plot(ax=plt.gca(),title='Spain')

plt.ylabel('Confirmed infection cases')

plt.subplot(3,2,4)

total_US.plot(ax=plt.gca(),title='US')

plt.ylabel('Confirmed infection cases')

plt.subplot(3,2,5)

total_India.plot(ax=plt.gca(),title='India')

plt.ylabel('Confirmed infection cases')





pop_china=1438086047

pop_italy=60481360

pop_spain= 46750731

pop_US=330568224

pop_india=1376937200

total_China.confirmed=total_China.confirmed/pop_china*100

total_Italy.confirmed=total_Italy.confirmed/pop_italy*100

total_Spain.confirmed=total_Spain.confirmed/pop_spain*100

total_US.confirmed=total_China.confirmed/pop_spain*100

total_India.confirmed=total_China.confirmed/pop_india*100

plt.figure(figsize=(15,10))

plt.subplot(3, 2, 1)

total_China.confirmed.plot(ax=plt.gca(), title='China')

plt.ylabel("Fraction of population infected")

plt.ylim(0, 0.5)

plt.subplot(3, 2, 2)

total_Italy.confirmed.plot(ax=plt.gca(), title='Italy')

plt.ylabel("Fraction of population infected")

plt.ylim(0, 0.5)

plt.subplot(3, 2, 3)

total_Spain.confirmed.plot(ax=plt.gca(), title='Spain')

plt.ylabel("Fraction of population infected")

plt.ylim(0, 0.5)

plt.subplot(3, 2, 4)

total_US.confirmed.plot(ax=plt.gca(), title='US')

plt.ylabel("Fraction of population infected")

plt.ylim(0, 0.5)

plt.subplot(3, 2, 5)

total_India.confirmed.plot(ax=plt.gca(), title='India')

plt.ylabel("Fraction of population infected")

plt.ylim(0, 0.5)


confirmed_total_italy=train[(train['country']=='Italy')&train['confirmed']!=0].groupby(['date']).agg({'confirmed':['sum']})

deaths_total_italy=train[(train['country']=='Italy') & train['confirmed']!=0].groupby(['date']).agg({'deaths':['sum']})

total_italy=confirmed_total_italy.join(deaths_total_italy)



confirmed_total_spain=train[(train['country']=='Spain')&train['confirmed']!=0].groupby(['date']).agg({'confirmed':['sum']})

deaths_total_spain=train[(train['country']=='Spain') & train['confirmed']!=0].groupby(['date']).agg({'deaths':['sum']})

total_spain=confirmed_total_spain.join(deaths_total_spain)



confirmed_total_us=train[(train['country']=='US')&train['confirmed']!=0].groupby(['date']).agg({'confirmed':['sum']})

deaths_total_us=train[(train['country']=='US') & train['confirmed']!=0].groupby(['date']).agg({'deaths':['sum']})

total_us=confirmed_total_italy.join(deaths_total_italy)



confirmed_total_india=train[(train['country']=='India')&train['confirmed']!=0].groupby(['date']).agg({'confirmed':['sum']})

deaths_total_india=train[(train['country']=='India') & train['confirmed']!=0].groupby(['date']).agg({'deaths':['sum']})

total_india=confirmed_total_india.join(deaths_total_india)







italy=[i for i in total_italy.confirmed['sum'].values]

italy_30=italy[0:80]



spain=[i for i in total_spain.confirmed['sum'].values]

spain_30=spain[0:80]



us=[i for i in total_us.confirmed['sum'].values]

us_30=us[0:80]



india=[i for i in total_india.confirmed['sum'].values]

india_30=india[0:80]

#plots

plt.figure(figsize=(14,8))



plt.plot(italy_30)

plt.plot(spain_30)

plt.plot(us_30)

plt.plot(india_30)

plt.legend(["Italy","Spain","US","India"],loc='upper_left')

plt.title("Covid-19 infections from the first confirmed case",size=15)

plt.xlabel("Days",size=13)

plt.ylabel("Infected cases",size=13)

plt.ylim(0,130000)

plt.show()



confirmed_total_italy=train[(train['country']=='Italy')&train['confirmed']!=0].groupby(['date']).agg({'confirmed':['sum']})

deaths_total_italy=train[(train['country']=='Italy') & train['confirmed']!=0].groupby(['date']).agg({'deaths':['sum']})

total_italy=confirmed_total_italy.join(deaths_total_italy)

italy=[i for i in total_italy.confirmed['sum'].values]

italy_30=italy[0:80]

plt.figure(figsize=(14,6))

plt.plot(italy_30)

plt.legend(["Italy"],loc='upper_left')

plt.title("Covid-19 infections from the first confirmed case",size=15)

plt.xlabel("Days",size=13)

plt.ylabel("Infected cases",size=13)

plt.ylim(0,130000)

plt.show()
from sklearn import preprocessing

from datetime import datetime

#merge train and test ,exclude overlap

dates_overlap=['2020-04-01','2020-04-02','2020-04-03','2020-04-04','2020-04-05','2020-04-06','2020-04-07','2020-04-08','2020-04-09','2020-04-10']

train2=train.loc[~train['date'].isin(dates_overlap)]

all_data=pd.concat([train2,test],axis=0,sort=False)

#Double check that there are no informed ConfirmedCases and deaths after 2020-03-11

all_data.loc[all_data['date']>='2020-04-01','confirmed']=0

all_data.loc[all_data['date']>='2020-04-01','deaths']=0

all_data['date']=pd.to_datetime(all_data['date'])

#create data columns

le=preprocessing.LabelEncoder()

all_data['day_num']=le.fit_transform(all_data.date)

all_data['day']=all_data['date'].dt.day

all_data['month']=all_data['date'].dt.month

all_data['Year']=all_data['date'].dt.year

#full null values given that we merged train-test datasets

all_data['province'].fillna("None",inplace=True)

all_data['confirmed'].fillna(0,inplace=True)

all_data['deaths'].fillna(0,inplace=True)

all_data['id'].fillna(-1,inplace=True)

all_data['forecastid'].fillna(-1,inplace=True)

display(all_data)

missings_count={col:all_data[col].isnull().sum() for col in all_data.columns}

missings=pd.DataFrame.from_dict(missings_count,orient='index')

print(missings.nlargest(30,0))
full_latest=all_data[all_data['day']==max(all_data['day'])]

fig=px.treemap(full_latest.sort_values(by='confirmed',ascending=False).reset_index(drop=True),path=["country"],values='confirmed',height=700,title="Number of Confirmed Cases",color_discrete_sequence=px.colors.qualitative.Dark2)

fig.data[0].textinfo='label+text+value'

fig.show()



fig=px.treemap(full_latest.sort_values(by='deaths',ascending=False).reset_index(drop=True),

              path=["country"],values='deaths',height=700,title="Number of Deaths Cases",color_discrete_sequence=px.colors.qualitative.Dark2)

fig.data[0].textinfo='label+text+value'

fig.show()
temp=train.groupby('date')['confirmed','deaths'].sum().reset_index()

temp=temp.melt(id_vars='date',value_vars=['confirmed','deaths'],var_name='Case',value_name='Count')



temp.head()

fig=px.area(temp,x="date",y="Count",color="Case",title="Cases Over Time",color_discrete_sequence=['#ffeebb',"#2367ff"])

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()