import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

font = {'family' : 'serif',

        'weight' : 'normal',

        'size'   : 18}

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

from wordcloud import WordCloud, STOPWORDS

from scipy.misc import imread

import base64



# Load in the .csv files as three separate dataframes

Global = pd.read_csv('../input/world-religions/global.csv') # Put to caps or else name clash

national = pd.read_csv('../input/world-religions/national.csv')

regional = pd.read_csv('../input//world-religions/regional.csv')
nat2=national[national['population']>10000000].sort_values(by='population', ascending=False)





nat=nat2[['year', 'state', 'code','christianity_percent', 'judaism_percent','islam_percent',

              'buddhism_percent','zoroastrianism_percent', 'hinduism_percent', 'sikhism_percent',

              'shinto_percent', 'baha’i_percent', 'taoism_percent', 'jainism_percent',

              'confucianism_percent', 'syncretism_percent', 'animism_percent',

              'noreligion_percent', 'otherreligion_percent']]#, 'religion_sumpercent']]



nat=nat[nat['year']==2010]

nat.drop(['year','code'],axis=1,inplace=True)

nat.set_index('state',inplace=True)

nat['main_percentage'] =nat.max(axis=1)

nat['main_religion']=nat.idxmax(axis=1)

nat['main_religion']=nat['main_religion'].astype(str)

nat['main_religion']=nat['main_religion'].apply(lambda x: x.split('_')[0])

x=nat.index

fig, ax = plt.subplots(figsize=(15,7),nrows=1, ncols=1,sharey=True)

colormap = plt.cm.terrain

nat.plot(kind='area',stacked=True,  colormap= 'tab20',grid=True, ax= ax , legend=True)

ax.legend(bbox_to_anchor=(-0.0, -0.4, 1, 0.1), loc=10,prop={'size':12},

           ncol=4, mode="expand", borderaxespad=0.)

ax.set_title('World Religions as Percentage of Population',y=1.02,size=25)

#ax.set_ylabel('Billions', color='gray')

ax.set_ylim(0,1)

ax.set_xticklabels(x, rotation = 45)

#ax.set_xticks(x, minor=True)

#plt.xticks(rotation=90) 

plt.show()
#nat.reset_index()



fig, ax = plt.subplots(figsize=(10,20),nrows=1, ncols=1,sharey=True)

colormap = plt.cm.terrain

nat2 = nat[30:]

nat2.plot(kind='barh',stacked=True,  colormap= 'tab20', width=.8,grid=True, ax= ax , legend=True)

ax.set_title('Country Religions as Percentage% of Population',size=25)

#ax.set_ylabel('Billions', color='gray')

#ax.set_ylim(0,1)

ax.set_xlim(0,1)

ax.legend(loc='upper right')

plt.show()
East_asian_countries = ['China', 'Mongolia', 'Taiwan', 'North Korea',

       'South Korea', 'Japan','Thailand', 'Cambodia',

       'Laos', 'Vietnam',  'Malaysia', 'Singapore',

       'Brunei', 'Philippines', 'Indonesia']



South_asian_countries = ['India', 'Bhutan', 'Pakistan', 'Bangladesh',

       'Sri Lanka', 'Nepal']



East_european_countries = [

    'Poland', 'Czechoslovakia', 'Czech Republic', 'Slovakia','Malta', 'Albania', 'Montenegro', 'Macedonia',

       'Croatia', 'Yugoslavia', 'Bosnia and Herzegovina', 'Kosovo',

       'Slovenia', 'Bulgaria', 'Moldova', 'Romania','Estonia', 'Latvia', 'Lithuania', 'Ukraine', 'Belarus',

       'Armenia', 'Georgia',

]



West_european_countries = [

    'United Kingdom', 'Ireland', 'Netherlands', 'Belgium', 'Luxembourg',

       'France', 'Liechtenstein', 'Switzerland', 'Spain', 'Portugal', 'Germany','Greece', 'Italy'

]



Africa = ['Mali', 'Senegal',

       'Benin', 'Mauritania', 'Niger', 'Ivory Coast', 'Guinea',

       'Burkina Faso', 'Liberia', 'Sierra Leone', 'Ghana', 'Togo',

       'Cameroon', 'Nigeria', 'Gabon', 'Central African Republic', 'Chad',

       'Congo', 'Democratic Republic of the Congo', 'Uganda', 'Kenya',

       'Tanzania', 'Burundi', 'Rwanda', 'Somalia']



South_america = ['Peru', 'Brazil',

       'Bolivia', 'Paraguay', 'Chile', 'Argentina', 'Uruguay','Colombia',

       'Venezuela']
sub_cat= [ 'christianity_protestant', 'christianity_romancatholic',

       'christianity_easternorthodox', 'christianity_anglican',

       'christianity_other', 'christianity_all', 'judaism_orthodox',

       'judaism_conservative', 'judaism_reform', 'judaism_other',

       'judaism_all', 'islam_sunni', 'islam_shi’a', 'islam_ibadhi',

       'islam_nationofislam', 'islam_alawite', 'islam_ahmadiyya',

       'islam_other', 'islam_all', 'buddhism_mahayana', 'buddhism_theravada',

       'buddhism_other', 'buddhism_all', 'zoroastrianism_all', 'hinduism_all',

       'sikhism_all', 'shinto_all', 'baha’i_all', 'taoism_all', 'jainism_all',

       'confucianism_all', 'syncretism_all', 'animism_all', 'noreligion_all',

       'otherreligion_all', 'religion_all', 'population', 'world_population']



main_cat = [ 'year', 'christianity_all','islam_all','noreligion_all','hinduism_all','syncretism_all',

            'buddhism_all','animism_all','shinto_all','otherreligion_all','judaism_all','sikhism_all',

            'taoism_all','confucianism_all','jainism_all','baha’i_all','zoroastrianism_all' ]

renam = {'christianity_all':'Christianity','islam_all':'Islam','noreligion_all':'No Religion',

         'hinduism_all': 'Hinduism','syncretism_all':'Syncretism','buddhism_all':'Buddhism',

         'animism_all':'Animism','shinto_all':'Shinto','otherreligion_all': 'Other Religion',

         'judaism_all':'Judaism','sikhism_all':'Sikhism','taoism_all':'Taoism',

         'confucianism_all':'Confucianism','jainism_all':'Jainism','baha’i_all':'Bahai',

         'zoroastrianism_all':'Zoroastrianism' }

world_rel = Global[main_cat].set_index('year')

world_rel = world_rel.rename(columns=renam)

fig, ax = plt.subplots(figsize=(12,7),nrows=1, ncols=1,sharey=True)

colormap = plt.cm.terrain

world_rel.plot(kind='area',stacked=True,  colormap= 'tab20', grid=True, ax= ax , legend=True)

ax.set_title('World Religions Adherent Growth',y=1.08,size=25)

ax.set_ylabel('Billions', color='gray')

ax.set_ylim(0,8000000000)

ax.legend(loc='upper left')

plt.show()
fig, axes = plt.subplots(figsize=(15,7),nrows=1, ncols=3,sharey=True)

colormap = plt.cm.YlGnBu



christianity_year = regional.groupby(['year','region']).christianity_all.sum()

christianity_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False,

                                 ax= axes[0] , legend=False)

axes[0].set_title('Christianity Adherents by Region',y=1.08,size=15)

axes[0].set_ylabel('Billions', color='gray')

axes[0].set_ylim(0,2200000000)



islam_year = regional.groupby(['year','region']).islam_all.sum()

islam_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, 

                          ax= axes[1], legend= False)

axes[1].set_title('Islam Adherents  by Region',y=1.08,size=15)

axes[1].legend(bbox_to_anchor=(-0.5, -0.2, 2, 0.1), loc=10,prop={'size':15},

           ncol=5, mode="expand", borderaxespad=0.)



hinduism_year = regional.groupby(['year','region']).hinduism_all.sum()

hinduism_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False,

                                 ax= axes[2] , legend=False)

axes[2].set_title('Hinduism Adherents  by Region',y=1.08,size=15)

axes[2].set_ylabel('Billions', color='gray') 



plt.tight_layout()

plt.show()
fig, axes = plt.subplots(figsize=(15,6),nrows=1, ncols=4,sharey=True)

colormap = plt.cm.YlGnBu



axes[0].set_ylim(0,1000000000)



buddhism_year = regional.groupby(['year','region']).buddhism_all.sum()

buddhism_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, 

                          ax= axes[0], legend= False)

axes[0].set_title('Buddhism Adherents by Region',y=1.08,size=15)

axes[0].set_ylabel('Hundred Millions', color='gray') 



shinto_year = regional.groupby(['year','region']).shinto_all.sum()

shinto_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, 

                          ax= axes[1], legend= False)

axes[1].set_title('Shinto Adherents by Region',y=1.08,size=15)

axes[1].legend(bbox_to_anchor=(-0.5, -0.2, 2, 0.1), loc=10,prop={'size':12},

           ncol=5, mode="expand", borderaxespad=0.)



noreligion_year = regional.groupby(['year','region']).noreligion_all.sum()

noreligion_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, 

                          ax= axes[2], legend= False)

axes[2].set_title('No Religions  by Region',y=1.08,size=15)



syncretism_year = regional.groupby(['year','region']).syncretism_all.sum()

syncretism_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, 

                          ax= axes[3], legend= False)

axes[3].set_title('Syncretism  by Region',y=1.08,size=15)





plt.tight_layout()

plt.show()
fig, axes = plt.subplots(figsize=(15,12),nrows=2, ncols=4,sharey=True)

colormap = plt.cm.YlGnBu



judaism_year = regional.groupby(['year','region']).judaism_all.sum()

judaism_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False,

                                 ax= axes[0,0] , legend=False)

axes[0,0].set_title('Judaism by Region',y=1.08,size=15)

axes[0,0].set_ylabel('Ten Millions', color='gray') 

axes[0,0].set_ylim(0,45000000)



sikhism_year = regional.groupby(['year','region']).sikhism_all.sum()

sikhism_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, 

                          ax= axes[0,1], legend= False)

axes[0,1].set_title('Sikhism by Region',y=1.08,size=15)





taoism_year = regional.groupby(['year','region']).taoism_all.sum()

taoism_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False,

                                 ax= axes[0,2] , legend=False)

axes[0,2].set_title('Taoism Region',y=1.08,size=15)

 

otherreligion_year = regional.groupby(['year','region'])['otherreligion_all'].sum()

otherreligion_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, 

                          ax= axes[0,3], legend= False)

axes[0,3].set_title('Other Religions',y=1.08,size=15)



jainism_year = regional.groupby(['year','region']).jainism_all.sum()

jainism_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False,

                                 ax= axes[1,0] , legend=False)

axes[1,0].set_title('Jainism by Region',y=1.08,size=15)

axes[1,0].set_ylabel('Ten Millions', color='gray') 

#axes[0].set_ylim(0,45000000)



confucianism_year = regional.groupby(['year','region']).confucianism_all.sum()

confucianism_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, 

                          ax= axes[1,1], legend= False)

axes[1,1].set_title('Confucianism by Region',y=1.08,size=15)

axes[1,1].legend(bbox_to_anchor=(0, -0.2, 2, 0.1), loc=10,prop={'size':12},

           ncol=5, mode="expand", borderaxespad=0.)



zoroastrianism_year = regional.groupby(['year','region']).zoroastrianism_all.sum()

zoroastrianism_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False,

                                 ax= axes[1,2] , legend=False)

axes[1,2].set_title('Zoroastrianism by Region',y=1.08,size=15)

 

bahai_year = regional.groupby(['year','region'])['baha’i_all'].sum()

bahai_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, 

                          ax= axes[1,3], legend= False)

axes[1,3].set_title('Baha’i by Region',y=1.08,size=15)





plt.tight_layout()

plt.show()
nat2.head()
nat2[['main_religion','main_percentage']]