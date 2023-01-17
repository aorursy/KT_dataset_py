# data manipulation 

import numpy as np 

import pandas as pd 

from scipy import stats



# visualizations

import seaborn as sns

import matplotlib.pyplot as plt



# Interactive visualizations

import plotly.plotly as py

import plotly.graph_objs as go

import plotly.tools as tls

from plotly.offline import iplot, init_notebook_mode

import cufflinks

import cufflinks as cf

import plotly.figure_factory as ff

from plotly import tools



# Using plotly + cufflinks in offline mode

init_notebook_mode(connected=True)

cufflinks.go_offline(connected=True)



import os

print(os.listdir("../input"))
df_dict = pd.read_csv("../input/Data_Dictionary.csv", sep=';')

df_brazil = pd.read_csv("../input/BRAZIL_CITIES.csv", sep=';')



## filtering the data to get just the col and rows that we are interested 

df_dict = df_dict[['FIELD', 'DESCRIPTION']][:81]
def resumetable(df, dict_):

    dict_ = dict_.rename(columns={'FIELD':'Name', 'DESCRIPTION': 'Description'})

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values

    summary = summary.merge(dict_, on='Name',how='left')

    for name in summary['Name'].value_counts().index:

        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

        

    return summary



def num_dist_plot(df, col_cat, col_num):

    print(f'Distribution of {col_cat}: ')

    tmp = df.groupby([col_cat])['CITY'].nunique().sort_index()

    tmp_pop = round((df_brazil.groupby([col_cat])[col_num].sum() / df_brazil.groupby([col_cat])[col_num].sum().sum()),2).sort_index() * 100

    

    trace1 = go.Bar(

        x=tmp.index, name='Total Cities',

        y=tmp.values, showlegend=False

    )



    trace2 =  go.Scatter(   

        x=tmp_pop.index,

        y=tmp_pop.values, yaxis='y2',

        name='%Population', opacity = 0.6, 

        marker=dict(

            color='black',

            line=dict(color='#000000',

                      width=2 )

        )

    )



    layout = dict(title =  f'Total Cities in each {str(col_cat)} and the % of Population by State',

              xaxis=dict(), 

              yaxis=dict(title= 'Count'), 

              yaxis2=dict(range= [0, 40], 

                          overlaying= 'y', 

                          anchor= 'x', 

                          side= 'right',

                          zeroline=False,

                          showgrid= False, 

                          title= '% of Total Population'

                         ))



    fig = go.Figure(data=[trace1, trace2], layout=layout)

    iplot(fig)



    

def cat_dist_plot(df, cat_group, cat_count):

    print(f'Distribution of {col_cat}: ')

    tmp = df_brazil.groupby(['STATE'])['REGIAO_TUR'].nunique()

    tmp_pop = round((df_brazil.groupby([col])['IBGE_RES_POP'].sum() / df_brazil.groupby([col])['IBGE_RES_POP'].sum().sum()),2).sort_index() * 100

    

    trace1 = go.Bar(

        x=tmp.index, name='Total Cities',

        y=tmp.values, showlegend=False

    )



    trace2 =  go.Scatter(   

        x=tmp_pop.index,

        y=tmp_pop.values, yaxis='y2',

        name='%Population', opacity = 0.6, 

        marker=dict(

            color='black',

            line=dict(color='#000000',

                      width=2 )

        )

    )



    layout = dict(title =  f'Total Cities in each {str(col)} and the % of Population by State',

              xaxis=dict(), 

              yaxis=dict(title= 'Count'), 

              yaxis2=dict(range= [0, 40], 

                          overlaying= 'y', 

                          anchor= 'x', 

                          side= 'right',

                          zeroline=False,

                          showgrid= False, 

                          title= '% of Total Population'

                         ))



    fig = go.Figure(data=[trace1, trace2], layout=layout)

    iplot(fig)
data_summary = resumetable(df_brazil, df_dict)



data_summary[:30]['Description'][1]
num_dist_plot(df_brazil, "STATE", 'IBGE_RES_POP')
IDH = ['IDHM', 'IDHM_Renda', 'IDHM_Longevidade', 'IDHM_Educacao']

GVA = ['GVA_AGROPEC', 'GVA_INDUSTRY', 'GVA_SERVICES', 'GVA_PUBLIC', ' GVA_TOTAL ']

EXPENDURE = ['TAXES', 'GDP', 'GDP_CAPITA']



for col in list(IDH+GVA):

    df_brazil[col] = df_brazil[col].str.replace(',', '.')

    df_brazil[col] = df_brazil[col].astype(float)
#First plot

trace0 = go.Box(

    x=df_brazil['STATE'],

    y=df_brazil['IDHM'], 

    showlegend=False, visible=True

)

#Second plot

trace1 = go.Box(

    x=df_brazil['STATE'],

    y=df_brazil['IDHM_Renda'], 

    showlegend=False, visible=False

)

#Second plot

trace2 = go.Box(

    x=df_brazil['STATE'],

    y=df_brazil['IDHM_Longevidade'], 

    showlegend=False, visible=False

)



#Third plot

trace3 = go.Box(

    x=df_brazil['STATE'],

    y=df_brazil['IDHM_Educacao'], 

    showlegend=False, visible=False

)

data = [trace0, trace1, trace2, trace3]





updatemenus = list([

    dict(active=0,

         x=-0.15,

         buttons=list([  

            dict(

                label = 'IDHM Distribution by State',

                 method = 'update',

                 args = [{'visible': [True, False, False, False]}, 

                     {'title': 'IDH by State'}]),

             

             dict(

                  label = 'IDHM Inc Distribution by State',

                 method = 'update',

                 args = [{'visible': [False, True, False, False]},

                     {'title': 'IDH Income by State'}]),



            dict(

                 label = 'IDHM Long Distribution by State',

                 method = 'update',

                 args = [{'visible': [False, False, True, False]},

                     {'title': 'IDH Longevity by State'}]),



            dict(

                 label =  'IDHM Educ Distribution by State',

                 method = 'update',

                 args = [{'visible': [False, False, False, True]},

                     {'title': 'IDH Education by State'}])

        ]),

    )

])



layout = dict(title='IDH METRICS by STATES (Select from Dropdown)', 

              showlegend=False,

              updatemenus=updatemenus)



fig = dict(data=data, layout=layout)



iplot(fig)
#fourth plot

trace4 = go.Box(

    x=df_brazil['STATE'],

    y=np.log(df_brazil[' GVA_TOTAL ']), name='GVA Total',

    showlegend=False, visible=True

)



#First plot

trace0 = go.Box(

    x=df_brazil['STATE'],

    y=np.log(df_brazil['GVA_AGROPEC']+1), name='Agropec',

    showlegend=False, visible=False

)

#Second plot

trace1 = go.Box(

    x=df_brazil['STATE'],

    y=np.log(df_brazil['GVA_INDUSTRY']+1), 

    showlegend=False, visible=False, name='Industry'

)

#Second plot

trace2 = go.Box(

    x=df_brazil['STATE'],

    y=np.log(df_brazil['GVA_SERVICES']+1), 

    showlegend=False, visible=False, name='Services'

)



#Third plot

trace3 = go.Box(

    x=df_brazil['STATE'],

    y=np.log(df_brazil['GVA_PUBLIC']+1), name='Public',

    showlegend=False, visible=False

)



data = [trace4, trace0, trace1, trace2, trace3, ]





updatemenus = list([

    dict(active=0,

         x=-0.15,

         buttons=list([  

            dict(

                label = 'Total GVA',

                 method = 'update',

                 args = [{'visible': [True, False, False, False, False]}, 

                     {'title': 'TOTAL GROSS VALUE ADDED (GVA) by STATE'}]),

             

            dict(

                label = 'Agropec GVA',

                 method = 'update',

                 args = [{'visible': [False, True, False, False, False]}, 

                     {'title': 'AGROPECUARY GROSS VALUE ADDED (GVA) by STATE'}]),

             dict(

                  label = 'Industry GVA',

                 method = 'update',

                 args = [{'visible': [False, False, True, False, False]},

                     {'title': 'INDUSTRY GROSS VALUE ADDED (GVA) by STATE'}]),



            dict(

                 label = 'Service GVA',

                 method = 'update',

                 args = [{'visible': [False, False, False, True, False]},

                     {'title': 'SERVICES GROSS VALUE ADDED (GVA) by STATE'}]),



            dict(

                 label =  'Public GVA',

                 method = 'update',

                 args = [{'visible': [False, False, False, False, True]},

                     {'title': 'PUBLIC GROSS VALUE ADDED (GVA) by STATE'}])

        ]),

    )

])



layout = dict(title='GVA VALUES by STATES (Select from Dropdown)', 

              showlegend=False,

              updatemenus=updatemenus)



fig = dict(data=data, layout=layout)



iplot(fig)
df_brazil.groupby('STATE')['REGIAO_TUR'].nunique().iplot(kind='bar', 

                                                         title='Total of Turistic Regions by State',

                                                         xTitle='State Name', yTitle='Count')

pd.crosstab(df_brazil['STATE'], df_brazil['CATEGORIA_TUR'], normalize='index').iplot(kind='bar', 

                                                                                     barmode='stack', 

                                                                                     title='% OF TURISM CATEGORY BY STATE',

                                                                                     xTitle='State Name', 

                                                                                     yTitle='% of Total')
pop = ['IBGE_1', 'IBGE_1-4', 'IBGE_5-9', 'IBGE_10-14', 'IBGE_15-59', 'IBGE_60+']

for col in pop:

    df_brazil[f'ratio_{col}'] = df_brazil[col] / df_brazil['IBGE_POP']
# Selecting the new categories and grouping by state and geting mean

age_ranges = ['STATE','ratio_IBGE_1-4', 'ratio_IBGE_5-9', 

              'ratio_IBGE_10-14', 'ratio_IBGE_15-59', 'ratio_IBGE_60+']



df_brazil[age_ranges].groupby('STATE').mean().iplot(kind='bar', barmode='stack', 

                                                    title='Mean of Age Ranges by State',

                                                    xTitle='State Name', yTitle='Mean % of Total Population')
list(data_summary.Name)
Compaines = ['COMP_A', 'COMP_B', 'COMP_C', 'COMP_D', 'COMP_E', 'COMP_F',

             'COMP_G', 'COMP_H', 'COMP_I', 'COMP_J', 'COMP_K', 'COMP_L', 

             'COMP_M', 'COMP_N', 'COMP_O', 'COMP_P', 'COMP_Q', 'COMP_R', 

             'COMP_S', 'COMP_T', 'COMP_U']



for col in Compaines:

    df_brazil[f'%_{col}'] = df_brazil[col] / df_brazil['COMP_TOT']

    

comp_ratio = ['%_COMP_A', '%_COMP_B', '%_COMP_C', '%_COMP_D', '%_COMP_E', 

              '%_COMP_F', '%_COMP_G', '%_COMP_H', '%_COMP_I', '%_COMP_J',

              '%_COMP_K', '%_COMP_L', '%_COMP_M', '%_COMP_N', '%_COMP_O', 

              '%_COMP_P', '%_COMP_Q', '%_COMP_R', '%_COMP_S', '%_COMP_T',

              '%_COMP_U','STATE']
df_brazil[comp_ratio].groupby('STATE').mean().iplot(kind='bar', barmode='stack', 

                                                    title='Mean of Age Ranges by State',

                                                    xTitle='State Name', yTitle='Mean % of Total Population')
pd.crosstab(df_brazil['STATE'], df_brazil['RURAL_URBAN'], normalize='index').iplot(kind='bar', 

                                                                                   barmode='stack',

                                                                                   title='RURAL OR URBAN % DISTRIBUTION BY STATE',

                                                                                   xTitle='State Names',

                                                                                   yTitle='Percent of Total Cities')