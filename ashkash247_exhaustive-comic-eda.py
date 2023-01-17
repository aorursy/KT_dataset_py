



import numpy as np # linear algebra

import pandas as pd # data p/rocessing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import plotly.graph_objs as go

import matplotlib.pyplot as plt

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

from plotly import tools

import warnings

warnings.filterwarnings('ignore')

import networkx as nx

%matplotlib inline

df_dc = pd.read_csv("../input/dc-wikia-data.csv")

df_marvel = pd.read_csv ("../input/marvel-wikia-data.csv")
df_top_10_Marvel=df_marvel.sort_values(by=['APPEARANCES'], ascending=False).head(10)



datamv10 = [

    go.Bar(

        x=df_top_10_Marvel['name'], 

        y=df_top_10_Marvel['APPEARANCES'],

         marker=dict(color=['rgba(222,45,38,0.8)', 

                            'rgba(10, 2, 150, 0.6)', 

                            'rgba(10, 2, 150, 0.6)', 

                            'rgba(10, 2, 150, 0.6)', 

                            'rgba(10,2,150,0.6)',

                             'rgba(10, 2, 150, 0.6)', 

                             'rgba(10, 2, 150, 0.6)', 

                             'rgba(10, 2, 150, 0.6)', 

                             'rgba(10, 2, 150, 0.6)',

                               'rgba(10, 2, 150, 0.6)'

                                                                            

                           

                           

                           ])

    )

]

layoutmv10 = go.Layout(

        margin=go.layout.Margin(

        l=100,

        r=100,

        b=200,

        t=100,

        pad=4

    ),

    autosize=False,

    width=900,

    height=700,

    title='Characters with highest number of appearances in Marvel comics',

    yaxis= dict(title='Frequency'),

     xaxis=dict(

        title='Characters',

        titlefont=dict(

            family='Arial, sans-serif',

            size=18,

            color='lightgrey'

        ),

        showticklabels=True,

        tickangle=30

)

)

fig1 = go.Figure(data=datamv10 , layout=layoutmv10)

iplot(fig1)



df_top_10_dc=df_dc.sort_values(by=['APPEARANCES'], ascending=False).head(10)



datadc10 = [

    go.Bar(

        x=df_top_10_dc['name'], 

        y=df_top_10_dc['APPEARANCES'],

        marker=dict(color=['rgba(0,0,0,0.8)', 

                            'rgba(100, 100, 100, 0.6)', 

                            'rgba(100, 100, 100, 0.6)', 

                            'rgba(100, 100, 100, 0.6)', 

                            'rgba(100,100,100,0.6)',

                             'rgba(100, 100, 100, 0.6)', 

                             'rgba(100, 100, 100, 0.6)', 

                             'rgba(100, 100, 100, 0.6)', 

                             'rgba(100, 100, 100, 0.6)',

                               'rgba(100, 100, 100, 0.6)'

                                                                            

                           

                           

                           ]),

    )

]

layoutdc10 = go.Layout(

        margin=go.layout.Margin(

        l=100,

        r=100,

        b=200,

        t=100,

        pad=4

    ),

    autosize=False,

    width=900,

    height=700,

    title='Characters with highest number of appearances in DC comics',

    yaxis= dict(title='Frequency'),

     xaxis=dict(

        title='Characters',

        titlefont=dict(

            family='Arial, sans-serif',

            size=18,

            color='lightgrey'

        ),

        showticklabels=True,

        tickangle=30

)

)

fig2 = go.Figure(data=datadc10 , layout=layoutdc10)

iplot(fig2)



df_marvel_bad= df_marvel[df_marvel['ALIGN']=="Bad Characters"]

df_Mbad_10 =df_marvel_bad.sort_values(by='APPEARANCES',ascending = False).head(10)

dataMbad10 = [

    go.Bar(

        x=df_Mbad_10['name'], 

        y=df_Mbad_10['APPEARANCES'],

           marker=dict(color=['rgba(34,139,34,0.8)', 

                            'rgba(100, 100, 100, 0.6)', 

                            'rgba(100, 100, 100, 0.6)', 

                            'rgba(100, 100, 100, 0.6)', 

                            'rgba(100,100,100,0.6)',

                             'rgba(100, 100, 100, 0.6)', 

                             'rgba(100, 100, 100, 0.6)', 

                             'rgba(100, 100, 100, 0.6)', 

                             'rgba(100, 100, 100, 0.6)',

                               'rgba(100, 100, 100, 0.6)'

                                                                            

                           

                           

                           ]),

    )

]

layoutMbad10 = go.Layout(

        margin=go.layout.Margin(

        l=50,

        r=100,

        b=200,

        t=100,

        pad=4

    ),

    autosize=False,

    width=900,

    height=700,

    title='Evil Characters with highest number of appearances in Marvel comics',

    yaxis= dict(title='Frequency'),

     xaxis=dict(

        title='Characters',

        titlefont=dict(

            family='Arial, sans-serif',

            size=18,

            color='lightgrey'

        ),

        showticklabels=True,

        tickangle=30

)

)

fig3 = go.Figure(data=dataMbad10 , layout=layoutMbad10)

iplot(fig3)
df_dc_bad= df_dc[df_dc['ALIGN']=="Bad Characters"]

df_dcbad_10 =df_dc_bad.sort_values(by='APPEARANCES',ascending = False).head(10)

data_dcbad10 = [

    go.Bar(

        x=df_dcbad_10  ['name'], 

        y=df_dcbad_10 ['APPEARANCES'],

        marker=dict(color=['rgba(34,139,34,0.8)', 

                            'rgba(89, 13, 89, 0.6)', 

                            'rgba(89, 13, 89, 0.6)', 

                            'rgba(89, 13, 89, 0.6)', 

                            'rgba(89,13,89,0.6)',

                             'rgba(89, 13, 89, 0.6)', 

                             'rgba(89, 13, 89, 0.6)', 

                             'rgba(89, 13, 89, 0.6)', 

                             'rgba(89, 13, 89, 0.6)',

                               'rgba(89, 13, 89, 0.6)'

                                                                            

                           

                           

                           ]))

]

layoutdcbad10 = go.Layout(

        margin=go.layout.Margin(

        l=50,

        r=100,

        b=200,

        t=100,

        pad=4

    ),

    autosize=False,

    width=900,

    height=700,

    title='Evil Characters with highest number of appearances in DC comics',

    yaxis= dict(title='Frequency'),

     xaxis=dict(

        title='Characters',

        titlefont=dict(

            family='Arial, sans-serif',

            size=18,

            color='lightgrey'

        ),

        showticklabels=True,

        tickangle=30

)

)

fig4 = go.Figure(data=data_dcbad10 , layout=layoutdcbad10)

iplot(fig4)



df_ID_freqM =df_marvel.groupby(['ID']).size().reset_index(name='counts')

#df_ID_freqM



df_ID_freqdc =df_dc.groupby(['ID']).size().reset_index(name='counts')

#df_ID_freqdc



trace1 =  go.Bar(

x= df_ID_freqM['ID'],

y=df_ID_freqM['counts'],

name = 'marvel')







trace2 = go.Bar(

x= df_ID_freqdc['ID'],

y=df_ID_freqdc['counts'],

name= 'DC')



fig5 = tools.make_subplots(rows=1, cols=2)



fig5.append_trace(trace1, 1, 1)

fig5.append_trace(trace2, 1, 2)



iplot(fig5)



df_marvel.EYE.value_counts()

df_dc.EYE.value_counts()

df_M_EYEfreq = df_marvel.groupby('EYE').size().reset_index(name='counts').sort_values('counts',ascending=False)









df_dc_EYEfreq =  df_dc.groupby('EYE').size().reset_index(name='counts').sort_values('counts',ascending=False)



dataMeye = [

    go.Bar(

    x= df_M_EYEfreq['EYE'],

    y=df_M_EYEfreq['counts'],

           marker=dict(

      color='rgba(10,10,120,0.7)'  )

)]

L1 = go.Layout(title='Frequency of eye colours in marvel characters')



fig6 = go.Figure(data=dataMeye,layout=L1)

iplot(fig6)





datadceye = [

    go.Bar(

    x= df_dc_EYEfreq['EYE'],

    y=df_dc_EYEfreq['counts'],

           marker=dict(

      color='rgba(10,10,120,0.7)' )

)]

L2 = go.Layout(title='Frequency of eye colours in DC characters')



fig6 = go.Figure(data=datadceye,layout=L2)

iplot(fig6)


datagrouped_m =  df_marvel.groupby('HAIR').size().reset_index(name='counts').sort_values(by='counts',ascending=False)

datagrouped_dc = df_dc.groupby('HAIR').size().reset_index(name='counts').sort_values(by='counts',ascending=False)



dataHM = [go.Bar(

x= datagrouped_m['HAIR'],

y= datagrouped_m['counts'],

       marker=dict(

      color='rgba(0,0,0,1)'  )

)]



l1 = go.Layout(title='Frequency of hair colours in marvel characters')



fig7 = go.Figure(dataHM,l1)









dataHdc = [go.Bar(

x= datagrouped_dc['HAIR'],

y= datagrouped_dc['counts'],

        marker=dict(

      color='rgba(0,0,0,1)'  )

)]



l2 = go.Layout(title='Frequency of hair colours in dc characters')



fig8 = go.Figure(dataHdc,l2)





iplot(fig7)

iplot(fig8)
df_marvel[df_marvel['SEX']=='Genderfluid Characters']

df_dc[df_dc['SEX']=='Transgender Characters']
plt.style.use('seaborn-whitegrid')



df_M_SO=df_marvel[ (-df_marvel['SEX'].isnull()) & (-df_marvel['GSM'].isnull())]

df_M_SO =  df_M_SO.groupby(['SEX','GSM']).size().reset_index(name= 'counts')



df_dc_SO=df_dc[ (-df_dc['SEX'].isnull()) & (-df_dc['GSM'].isnull())]

df_dc_SO =  df_dc_SO.groupby(['SEX','GSM']).size().reset_index(name= 'counts')



plt.figure(figsize=(15,15))

plt.subplot(2,1,1)

sns.barplot(data=df_M_SO,x=df_M_SO.SEX,y=df_M_SO.counts,hue=df_M_SO.GSM)

plt.title("Sexual orientations of Marvel characters across genders")







plt.subplot(2,1,2)

sns.barplot(data=df_dc_SO,x=df_dc_SO.SEX,y=df_dc_SO.counts,hue=df_dc_SO.GSM)

plt.title("Sexual orientations of DC characters across genders")

plt.show()

plt.style.use('fivethirtyeight')

df_marvel_IDA = df_marvel.groupby(['ID','ALIGN']).size().reset_index(name='counts')

df_dc_IDA  = df_dc.groupby(['ID','ALIGN']).size().reset_index(name='counts')



plt.figure(figsize=(15,15))

plt.subplot(2,1,1)

sns.barplot(x=df_marvel_IDA.ID,y=df_marvel_IDA.counts,hue=df_marvel_IDA.ALIGN,palette=["red", "green","orange"])

plt.title("Frequency of Marvel characters across identity and alignment")



g=plt.subplot(2,1,2)

sns.barplot(x=df_dc_IDA.ID,y=df_dc_IDA.counts,hue=df_dc_IDA.ALIGN,palette=["red", "orange","green","yellow"]  )

g.legend(loc='upper left')

plt.title("Frequency of DC characters across identity and alignment")

plt.show()

dfN=df_marvel[(df_marvel['SEX']=='Female Characters') & (df_marvel['GSM']=='Homosexual Characters')]

dfN['act'] ="Marvel female Homosexuals" 

G=nx.from_pandas_edgelist(dfN, 'act', 'name')

plt.figure(figsize=(10,10))

nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, width=1.0, edge_cmap=plt.cm.Blues)

plt.title("Female Homosexuals in the marvel dataset")

plt.show()

dfdcHF=df_dc[(df_dc['SEX']=='Female Characters') & (df_dc['GSM']=='Homosexual Characters')]

dfdcHF['act'] ="DC female Homosexuals" 

G2=nx.from_pandas_edgelist(dfdcHF, 'act', 'name')

plt.figure(figsize=(10,10))

nx.draw(G2, with_labels=True, node_color='skyblue', node_size=1500, width=1.0, edge_cmap=plt.cm.Blues)

plt.title("Female Homosexuals in the DC dataset")

plt.show()
df_MAA = df_marvel.groupby(['ALIGN','ALIVE']).size().reset_index(name='counts')

df_dcAA = df_dc.groupby(['ALIGN','ALIVE']).size().reset_index(name='counts')

plt.figure(figsize=(15,15))

plt.subplot(211)

sns.barplot(x=df_MAA.ALIGN,y=df_MAA.counts,hue=df_MAA.ALIVE)

plt.title("Distribuition of Marvel charcaters across Alignment and Mortality")



plt.subplot(212)

sns.barplot(x=df_dcAA.ALIGN,y=df_dcAA.counts,hue=df_dcAA.ALIVE)

plt.title("Distribuition of DC charcaters across Alignment and Mortality")