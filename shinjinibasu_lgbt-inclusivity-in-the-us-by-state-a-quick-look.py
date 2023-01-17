import numpy as np 

import pandas as pd 

import os

#print(os.listdir("../input"))



%matplotlib inline



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs,init_notebook_mode, plot, iplot

#init_notebook_mode(connected=True)



sns.set()
df_2017 = pd.read_csv("../input/us-census-demographic-data/acs2015_census_tract_data.csv")

df_2015 = pd.read_csv("../input/us-census-demographic-data/acs2015_census_tract_data.csv")

lgbt = pd.read_csv('../input/map-lgbt-policy-tally/MAP.csv')

ideo = pd.read_csv('../input/state-ideology/stateideology_v2018.csv',header=None)

arda = pd.read_csv('../input/arda-state/ARDA_State.csv')
lgbt.info()
lgbt.head()
lgbt.describe()
id_col = ['State', 'State_id','Year','Citi_ideo','Govt_ideo']



ideo.columns = id_col
ideo.isna().sum()
print(ideo.loc[ideo.State.isna()])





ideo.loc[ideo.Citi_ideo.isna()]
ideo = ideo.sort_values(['State_id','Year']).fillna(method='ffill')
ideo_recent = ideo.loc[ideo['Year']>=2015].groupby('State',as_index=False)['State_id','Citi_ideo','Govt_ideo'].mean()



ideo_recent.head()
arda_col = ['STNAME']

for col in arda.columns:

    if 'RATE' in col:

        arda_col.append(col)

arda_col
arda_rel = arda[arda_col[:9]]



arda_rel['STID'] = lgbt.loc[lgbt['State']==arda['STNAME'],['State_id']]

arda_rel.fillna(0,inplace=True)

arda_rel.head()
for col in arda_rel.columns:

    if col in ['STNAME','STID','TOTRATE']:

        continue

    else:

        arda_rel[col] = arda_rel[col]/arda_rel['TOTRATE']



arda_rel.head()
full = [df_2017,df_2015]
for df in full:

    df['Other Race'] = 100 - df['White']-df['Black']-df['Hispanic']-df['Asian']-df['Native']-df['Pacific']
df_2017.columns
rel_col = ['State','CensusTract','TotalPop','Men','Women','White', 'Black', 'Hispanic', 'Asian', 'Pacific', 'Native', 'Other Race',

           'Income','IncomePerCap','Poverty','Unemployment']
df_2017 = df_2017[rel_col]

df_2015 = df_2015[rel_col]
def by_state(df):

    

    df1 = lgbt.copy()[['State', 'State_id']]

        

    for col in df.columns:

        if col in ['CensusTract','State']:

            continue

        if 'TotalPop' in col:

            df1[col] = df.groupby(['State'],as_index=False)[col].sum()[col]

        else:

            df1[col+'_rate'] = df.groupby(['State'],as_index=False)[col].sum()[col].divide(df1['TotalPop'])

        

    return df1





def clean(df):

    df2 = df.copy()    

    for col in df.columns:

        

        if col in ['CensusTract','State','TotalPop','Men','Women']:

            continue

        

        df2[col]=df2[col].mul(df2['TotalPop'],fill_value=1)

        

        dfnew = by_state(df2)

    return dfnew
df17_state=clean(df_2017)



df17_state.head()
init_notebook_mode(connected=True)



so_scale = [

    [0.0, 'rgb(242,240,247)'],

    [0.2, 'rgb(218,218,235)'],

    [0.4, 'rgb(188,189,220)'],

    [0.6, 'rgb(158,154,200)'],

    [0.8, 'rgb(117,107,177)'],

    [1.0, 'rgb(84,39,143)']

]



gi_scale = [

    [0.0, 'rgb(247,240,242)'],

    [0.2, 'rgb(235,218,218)'],

    [0.4, 'rgb(220,189,188)'],

    [0.6, 'rgb(200,154,158)'],

    [0.8, 'rgb(177,107,117)'],

    [1.0, 'rgb(143,39,84)']]

    

so_data = [dict(type='choropleth',

    colorscale = so_scale,

    autocolorscale = False,

    locationmode = 'USA-states',

    locations = lgbt['State_id'],

    z = lgbt['SO_Tally'],

    colorbar = go.choropleth.ColorBar()

)]



so_layout = dict(

    title = 'Sexual Orientation Tally',

    geo = dict(

        scope = 'usa',

        projection = {'type':'albers usa'},

        showlakes = True,

        lakecolor = 'rgb(255, 255, 255)')

)



fig1 = go.Figure(data = so_data, layout = so_layout)

iplot(fig1)



gi_data = [dict(type='choropleth',

    colorscale = gi_scale,

    autocolorscale = False,

    locationmode = 'USA-states',

    locations = lgbt['State_id'],

    z = lgbt['GI_Tally'],

    colorbar = go.choropleth.ColorBar()

)]



gi_layout = dict(

    title = 'Gender Identity Tally',

    geo = dict(

        scope = 'usa',

        projection = {'type':'albers usa'},

        showlakes = True,

        lakecolor = 'rgb(255, 255, 255)')

)



fig2 = go.Figure(data = gi_data, layout = gi_layout)

iplot(fig2)

data = [dict(type='choropleth',

    colorscale = 'Viridis',

    autocolorscale = False,

    reversescale=True,

    locationmode = 'USA-states',

    locations = lgbt['State_id'],

    z = lgbt['Tot_Tally'],

    text = lgbt['State_id'].astype(str),

    colorbar = go.choropleth.ColorBar()

)]



layout = dict(

    title = 'Total Policy Tally',

    geo = dict(

        scope = 'usa',

        projection = {'type':'albers usa'},

        showlakes = True,

        lakecolor = 'rgb(255, 255, 255)')

)



fig = go.Figure(data = data, layout = layout)

iplot(fig)

ideo_recent.head()
_ = ideo_recent.plot(kind='bar',x='State',y=['Citi_ideo','Govt_ideo'],stacked=True,color=['darkblue','skyblue'],figsize = (20,8),

                     legend=None)

_ = plt.xticks(rotation=90)

_ = plt.xlabel('State')

_ = plt.ylabel('Ideology')

_ = plt.legend(('Citizen Idelogy','Govt. Ideology'))
ideo_recent['Tot_ideo'] = ideo_recent['Citi_ideo']+ideo_recent['Govt_ideo'] 



ideo_recent.sort_values(['Tot_ideo'],ascending=False).head(10).plot(kind='bar',x='State',y=['Citi_ideo','Govt_ideo'],

                                                                    figsize=(12,6),stacked=True,color=['darkblue','lightblue'],legend=None)

plt.xticks(rotation=60)

plt.xlabel('State')

plt.ylabel('Ideology')

plt.legend(('Citizen Idelogy','Govt. Ideology'))

plt.title('States With Highest Ideology Score')

ideo_recent.sort_values(['Tot_ideo','Citi_ideo','Govt_ideo'],ascending=False).tail(10).plot(kind='bar',x='State',y=['Citi_ideo','Govt_ideo'],

                                                                                            figsize=(12,6),stacked=True,color=['darkblue','lightblue'],legend=None)

plt.xticks(rotation=60)

plt.xlabel('State')

plt.ylabel('Ideology')

plt.legend(('Citizen Idelogy','Govt. Ideology'))

plt.title('States With Lowest Ideology Score')

_ = lgbt.plot(kind='bar',x='State_id',y=['SO_Tally','GI_Tally'],stacked=True,color=['navy','lightseagreen'],figsize = (20,8),

                     legend=None)

_ = plt.xticks(rotation=60)

_ = plt.xlabel('State')

_ = plt.ylabel('Policy Tally')

_ = plt.legend(('Sexual Orientation Policy','Gender Identity Policy'),loc ='upper right')
lgbt.sort_values(['Tot_Tally'],ascending=False).head(10).plot(kind='bar',x='State',y=['SO_Tally','GI_Tally'],

                                                              figsize=(12,6),stacked=True,color=['navy','lightseagreen'],legend=None)

plt.xticks(rotation=60)

plt.xlabel('State')

plt.ylabel('Policy Tally')

plt.legend(('Sexual Orientation Tally','Gender Identity Tally'))

plt.title('States With Highest Policy Tally')

lgbt.sort_values(['Tot_Tally','GI_Tally','SO_Tally'],ascending=True).head(10).plot(kind='bar',x='State',y=['SO_Tally','GI_Tally'],

                                                                                            figsize=(12,6),stacked=True,color=['navy','lightseagreen'],legend=None)

plt.xticks(rotation=60)

plt.xlabel('State')

plt.ylabel('Policy Tally')

plt.legend(('Sexual Orientation Tally','Gender Identity Tally'))

plt.title('States With Lowest Policy Tally')

full = df17_state.merge(lgbt,on=['State','State_id'],how='inner').merge(arda_rel,left_on=['State','State_id'],right_on=['STNAME','STID'],how='inner').drop(

    ['STNAME','STID','TOTRATE'],axis=1)

full.head()
full.drop(['Other Race_rate','Income_rate','Men_rate','Women_rate'],axis=1,inplace=True)

data = full.merge(ideo_recent.drop('State_id',axis=1),on=['State'])
corr = data.corr()

cmap = sns.cubehelix_palette(8,light=1, as_cmap=True)

_ , ax = plt.subplots(figsize =(14, 10))

hm = sns.heatmap(corr, ax= ax, annot= True,linewidths=0.2,cmap=cmap)