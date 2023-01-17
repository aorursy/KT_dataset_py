import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
heros = pd.read_csv('../input/heroes_information.csv')

powers = pd.read_csv('../input/super_hero_powers.csv')
heros.head()
powers.head()
print(heros.shape)

print(powers.shape)
heros.loc[heros['name']== 'Superman']['Alignment']
heros.loc[heros['name'] == 'Thanos']['Alignment']
heros['Alignment'].unique()
sum(heros['name'].isna())
heros.drop(['Unnamed: 0'], axis = 1, inplace = True)
heros.head()
heros.info()
powers.info()
sum(heros['Publisher'].isna())
heros.replace('-','unknown',inplace=True)
heros.info()
heros['Weight'].value_counts()
#relace the -99 to nan

heros.replace(-99,np.nan,inplace=True)

heros['Weight'].value_counts()
heros['Weight'].isna().sum()
ht_wt = heros[['Weight','Height']]
ht_wt.head(2)
#applay the imputer

from sklearn.impute import SimpleImputer



imputer = SimpleImputer(strategy = 'median')

X = imputer.fit_transform(ht_wt)

hero_h_w = pd.DataFrame(X,columns = ht_wt.columns)
hero_h_w.head()
hero_h_w.isna().sum()
heros_wo_h_w = heros.drop(['Weight','Height'], axis = 1)
heros_wo_h_w.head()
heros = pd.concat([heros_wo_h_w,hero_h_w],axis =1)
heros.head(5)
heros['Publisher'].value_counts()
publisher_series = heros['Publisher'].value_counts()

publishers = list(publisher_series.index)

publications = list((publisher_series/publisher_series.sum())*100)
colors = sns.color_palette()
plt.pie(publications,labels=publishers,colors=colors,autopct='%1.2f%%')

fig = plt.gcf()

fig.set_size_inches(10,8)

plt.show()
import plotly as py

import plotly.graph_objs as go

from plotly import tools

#py.init_notebook_mode(connected =True)
draw = go.Pie(labels=publishers, values=publications)



layout = go.Layout(title ='%age publications by publishers',height = 600,width = 600)

data = [draw]

fig = go.Figure(data = data,layout=layout)

py.offline.iplot(fig,filename="publications-by-publishers")
heros.loc[heros['name']=='Deadpool']
heros['Alignment'].unique()
df = pd.DataFrame(columns=['Publisher','total-heros','total-villans','total-nuetral','total-unknow'])



for publisher in publishers:

    data =[]

    data.append(publisher)

    data.append(len(heros.loc[(heros['Alignment'] == 'good') & (heros['Publisher']==publisher),'name']))

    data.append(len(heros.loc[(heros['Alignment'] == 'bad') & (heros['Publisher']==publisher),'name']))

    data.append(len(heros.loc[(heros['Alignment'] == 'neutral') & (heros['Publisher']==publisher),'name']))

    data.append(len(heros.loc[(heros['Alignment'] == 'unknown') & (heros['Publisher']==publisher),'name']))

    

    df.loc[len(df)] = data
df.head(5)
N = len(df)

idx = np.arange(N)

color_list = ['G','R','B','O']

gap = 0.35



plt.bar(idx,df['total-heros'], label = 'Heros', width = gap)

plt.bar(idx,df['total-villans'], label = 'Villans', width = gap)

plt.bar(idx,df['total-nuetral'], label = 'Nuetral', width = gap)

plt.bar(idx,df['total-unknow'], label = 'Unknown', width = gap)

plt.legend()



plt.show()
#using plotly

block1 = go.Bar(x = list(df['Publisher']), y = list(df['total-heros']),name = 'total-heros')

block2 = go.Bar(x = list(df['Publisher']), y = list(df['total-villans']),name = 'total-villans')

block3 = go.Bar(x = list(df['Publisher']), y = list(df['total-nuetral']),name = 'total-nuetral')

block4 = go.Bar(x = list(df['Publisher']), y = list(df['total-unknow']),name = 'total-unknown')



data_obj = [block1,block2,block3,block4]

layout = go.Layout(title='count of character allignment',barmode='group')

fig = go.Figure(data = data_obj, layout = layout)

py.offline.iplot(fig,filename='bar')
gender_series = heros['Gender'].value_counts()

genders = list(gender_series.index)

distribution = list((gender_series/gender_series.sum())*100)



draw = go.Pie(labels = genders, values = distribution)

layout = go.Layout(title='gender wise distriburion of supper heros',height =600, width =600)

data_obj = [draw]

fig = go.Figure(data = data_obj, layout=layout)



py.offline.iplot(fig,filename='gender disrtibutiongender wise distriburion of supper heros')
heros_gender_series = heros['Gender'].loc[heros['Alignment'] =='good'].value_counts()

heros_gender = list(heros_gender_series.index)

heros_distribution = list((heros_gender_series/heros_gender_series.sum())*100)



villans_gender_series = heros['Gender'].loc[heros['Alignment'] =='bad'].value_counts()

villans_gender = list(heros_gender_series.index)

villans_distribution = list((heros_gender_series/heros_gender_series.sum())*100)



neutrals_gender_series = heros['Gender'].loc[heros['Alignment'] =='neutrals'].value_counts()

neutrals_gender = list(heros_gender_series.index)

neutrals_distribution = list((heros_gender_series/heros_gender_series.sum())*100)



unknown_gender_series = heros['Gender'].loc[heros['Alignment'] =='unknown'].value_counts()

unknown_gender = list(heros_gender_series.index)

unknown_distribution = list((heros_gender_series/heros_gender_series.sum())*100)

fig ={

    'data':[

        {

                'labels':heros_gender,

                'values':heros_distribution,

                'type':'pie',

                'hole':0.4

               #"name":'heros',

                #'domain':{'row'=0,'coloumn'=0}

        },

        {

            'labels':villans_gender,

            'values':villans_distribution,

            'type':'pie',

            'hole':0.4

            #'name':'villans',

            #'domain':{'row'=0,'coloumn'=1}

        },

         {

            'labels':neutrals_gender,

            'values':neutrals_distribution,

            'type':'pie',

            'hole':0.4

           #'name':'nuetrals',

            #'domain':{'row'=1,'coloumn'=0}

        },

        {

            'labels':unknown_gender,

            'values':unknown_distribution,

            'type':'pie',

            'hole':0.4

            #'name':'nuetrals',

           # 'domain':{'row'=1,'coloumn'=1}

        }

    ],

    'layout':{

        'title':'Gender distributions by alignment',

        'grid' : {'rows':2, 'columns':2},

        'height':650,

        'width':650

    }

}

py.offline.iplot(fig,filename='Gender distributions by alignment')
#Bar Graph

male_df = heros.loc[heros['Gender']=='Male']

female_df = heros.loc[heros['Gender']=='Female']
trace_m = go.Bar(

    x = male_df['Alignment'].value_counts().index,

    y = male_df['Alignment'].value_counts().values,

   name='Male'

)



trace_f = go.Bar(

    x = female_df['Alignment'].value_counts().index,

    y = female_df['Alignment'].value_counts().values,

    name= 'female'

)



data_obj = [trace_m,trace_f]

layout = go.Layout(title='characters by their gender by alignment',barmode = 'group')

fig = go.Figure(data = data_obj,layout=layout)

py.offline.iplot(fig,filename='characters by their gender by alignment')
heros['Race'].unique()
trace = go.Bar(

    x = heros['Race'].value_counts().index,

    y = heros['Race'].value_counts().values,

    name="Races"

)



layout = go.Layout(

    title="distribution across different races"

)



fig = go.Figure(data=[trace], layout=layout)

py.offline.iplot(fig, filename='distribution across different races')
heros['Hair color'].unique()
# distribution of bald and chacarters with hair



heros['bald_or_not'] = heros['Hair color'].where(heros['Hair color']=='No Hair', other='Hair')



heros.head()
trace = go.Bar(

    x = heros['bald_or_not'].value_counts().index,

    y = heros['bald_or_not'].value_counts().values,

    name='bald vs not bald',

    text=['not bald', 'bald']

)



layout = go.Layout(

    title = 'bald vs not bald'

)



fig = go.Figure(data=[trace], layout=layout)

py.offline.iplot(fig, filename='bald vs not bald')
powers.head()
powers = powers * 1

powers.head(2)
powers.loc[:,'total_powers'] = powers.iloc[:, 1:].sum(axis=1)

powers.head()
# most powerfull superhero



powers.sort_values(by='total_powers', ascending=False).head()
# using seaborn



plt.figure(figsize=(15,10))

sns.barplot(powers['hero_names'], powers['total_powers'], alpha=1)

plt.title("total powers by characters", fontsize=20)

plt.xticks(rotation=90)

plt.ylabel("total powers", fontsize=14)

plt.xlabel("comic characters", fontsize=14)

plt.show()
# using plotly



trace = go.Bar(

    x=powers['hero_names'],

    y=powers['total_powers'],

    text = ['names','total_powers']

)



layout = go.Layout(

    title = "comic character by total powers they have"

)



fig = go.Figure(data=[trace], layout=layout)

py.offline.iplot(fig, filename="most powerful superhero")
top_30_powerful = powers.sort_values('total_powers', ascending=False).head(30)



trace = go.Bar(

    x = top_30_powerful['hero_names'],

    y = top_30_powerful['total_powers'],

    text = ['names', 'total_powers']

)



layout =go.Layout(

    title="top 30 most powerful hero"

)



fig = go.Figure(data=[trace], layout=layout)

py.offline.iplot(fig, filename="top 30")