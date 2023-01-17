import pandas as pd

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

%matplotlib inline



import plotly

import plotly.express as px

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

import plotly.figure_factory as ff

from plotly import subplots

from plotly.subplots import make_subplots

init_notebook_mode(connected=True)



from datetime import date, datetime, timedelta

import time, re, os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



def resumetable(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values



    return summary



df = pd.read_csv('/kaggle/input/2020-democratic-primary-endorsements/endorsements-2020.csv')

df.head(10)
df.rename(columns={'endorser party': 'party'}, inplace=True)

resumetable(df)
percent_missing = np.round(df.isnull().sum() * 100 / len(df),2)

missing_value_df = pd.DataFrame({'column_name': df.columns,

                                 'percent_missing': percent_missing}).sort_values('percent_missing', ascending=False)





fig = go.Figure()

fig.add_trace(

        go.Bar(x=missing_value_df['column_name'],

               y=missing_value_df['percent_missing'],

               opacity=0.9,

               text=missing_value_df['percent_missing'],

               textposition='inside',

               marker={'color':'indianred'}

                   ))

fig.update_layout(

      title={'text': 'Percentage Missing by Column',

             'y':0.95, 'x':0.5,

            'xanchor': 'center', 'yanchor': 'top'},

      showlegend=False,

      xaxis_title_text='Columns',

      yaxis_title_text='Percentage',

      bargap=0.1

    )



fig.show()
fig = go.Figure(

        go.Heatmap(

            z=df.isnull().astype(int),

            x=df.columns,

            y=df.index.values,

            colorscale='Greys',

            reversescale=True,

            showscale=False))



fig.update_layout(

    title={'text': 'Missing values Matrix',

             'y':0.95, 'x':0.5,

            'xanchor': 'center', 'yanchor': 'top'},

    xaxis=dict(tickangle=45, title='Columns'),

    yaxis=dict(range=[np.max(df.index.values), np.min(df.index.values)], title='Row Index'),

    )

fig.show()



df.drop(['city', 'body', 'order', 'district'], axis=1, inplace=True)

((df[['source', 'date', 'endorsee']].isnull()).astype(int).sum(axis=1)).value_counts()
df.rename(columns={'source': 'raw_source'}, inplace=True)

df['raw_source'] = df.loc[:,'raw_source'].fillna('other')

df['source'] = 'other'



keys=['twitter', 'politico', 'youtube', '4president', 'cnn', 'apnews']



for k in keys:

    df['source'] =  np.where(df['raw_source'].str.contains(k), k,  df['source'])

    

df.drop('raw_source', axis=1, inplace=True)

df['endorsee'] = df.loc[:,'endorsee'].fillna('no_endorsee')

df['party'] = df.loc[:, 'party'].fillna('None')

resumetable(df)
state_to_s = {

 'Alabama': 'AL',

 'Alaska':'AK',

 'Arizona':'AZ',

 'Arkansas':'AR',

 'California':'CA',

 'Colorado':'CO',

 'Connecticut':'CT',

 'Delaware':'DE',

 'Florida':'FL',

 'Georgia':'GA',

 'Hawaii':'HI',

 'Idaho':'ID',

 'Illinois':'IL',

 'Indiana':'IN',

 'Iowa':'IA',

 'Kansas':'KS',

 'Kentucky':'KY',

 'Louisiana':'LA',

 'Maine':'ME',

 'Maryland':'MD',

 'Massachusetts':'MA',

 'Michigan':'MI',

 'Minnesota':'MN',

 'Mississippi':'MS',

 'Missouri':'MO',

 'Montana':'MT',

 'Nebraska':'NE',

 'Nevada':'NV',

 'New Hampshire':'NH',

 'New Jersey':'NJ',

 'New Mexico':'NM',

 'New York':'NY',

 'North Carolina' :'NC',

 'North Dakota':'ND',

 'Ohio':'OH',

 'Oklahoma':'OK',

 'Oregon':'OR',

 'Pennsylvania':'PA',

 'Rhode Island':'RI',

 'South Carolina':'SC',

 'South Dakota':'SD',

 'Tennessee':'TN',

 'Texas':'TX',

 'Utah':'UT',

 'Vermont':'VT',

 'Virginia':'VA',

 'Washington':'WA',

 'West Virginia':'WV',

 'Wisconsin':'WI',

 'Wyoming':'WY',

 'District of Columbia':'DC',

 'Marshall Islands':'MH'}



s_to_state = {}



for k,v in state_to_s.items():

    s_to_state[v]=k

    

df['full_state'] = df['state'].map(s_to_state)
endorsee_df = df[df['endorsee']!='no_endorsee']

endorsee_df['endorsee'] = endorsee_df['endorsee'].str.split(' ').apply(lambda r: r[-1])

endorsee_df.head(10)
end_df = endorsee_df.groupby('endorsee').agg({'endorser': 'count', 'points': 'sum'})



end_df.rename(columns={'endorser': 'n_endorsements',

                       'points': 'tot_points'},

              inplace=True)



end_df['points_endorser_ratio'] = np.round(np.divide(end_df['tot_points'].to_numpy(), end_df['n_endorsements'].to_numpy()), 2)

end_df.reset_index(inplace=True)
fig = go.Figure()



fig.add_trace( 

        go.Scatter(

            x=end_df['n_endorsements'], 

            y=end_df['tot_points'],

            mode='markers+text',

            marker=dict(

                size=(end_df['points_endorser_ratio']+3)**2,

                color=end_df["points_endorser_ratio"],

                colorscale='geyser',

                opacity = 0.7),

            text=end_df['endorsee'],

            textposition='bottom right'

    ))



fig.update_layout(

        xaxis_type="log",

        yaxis_type="log",

        title={'text': 'Total Points per Number of Endorsers',

               'y':0.95, 'x':0.5,

               'xanchor': 'center', 'yanchor': 'top'},

        showlegend=False,

        xaxis_title_text='Number of Endorsers',

        yaxis_title_text='Total Points',

        updatemenus = list([

            dict(active=0,

                 buttons=list([

                    dict(label='Log Scale',

                         method='update',

                         args=[{'visible': True},

                               {'title': 'Log scale',

                                'xaxis': {'type': 'log'},

                                'yaxis': {'type': 'log'}}]),

                    dict(label='Log X',

                         method='update',

                         args=[{'visible': True},

                               {'title': 'Linear scale',

                                'xaxis': {'type': 'log'},

                                'yaxis': {'type': 'linear'}}]),

                    dict(label='Log Y',

                        method='update',

                       args=[{'visible': True},

                              {'title': 'Linear scale',

                               'xaxis': {'type': 'linear'},

                               'yaxis': {'type': 'log'}}]),

                    dict(label='Linear Scale',

                        method='update',

                       args=[{'visible': True},

                              {'title': 'Linear scale',

                               'xaxis': {'type': 'linear'},

                               'yaxis': {'type': 'linear'}}]),

                            ]),

                direction="down",

                pad={"r": 10, "t": 10},

                showactive=True,

                x=-0.2,

                xanchor="left",

                y=1.1,

                yanchor="top"

                )]),

        annotations=[

            go.layout.Annotation(text="Select Axis Scale", 

                                 x=-0.2, xref="paper", 

                                 y=1.13, yref="paper",

                                 align="left", showarrow=False),

        ])



fig.show()
cols = ['category', 'source', 'position', 'party', 'state']

lc = len(cols)



d={}



for c in cols:

    tmp = endorsee_df.groupby(['endorsee', c]).agg({'points':'sum', 'endorser':'count'}).reset_index()

    tmp.rename(columns={'points': f'pt_by_{c}', 'endorser': f'votes_by_{c}'}, inplace=True)

    d[c] = tmp



cat_df = d['category']

source_df = d['source']

position_df = d['position']

party_df = d['party']

state_df = d['state']

state_df['full_state'] = state_df['state'].map(s_to_state)



buttons=[]

l=endorsee_df['endorsee'].nunique()

n_plots=5

colors = ['cadetblue', 'indianred',  'goldenrod']

pie_colors = [ 'mediumpurple', 'beige']
fig = make_subplots(

    rows=3, cols=2,

    specs=[[{'colspan':2}, None],

           [{}, {"type": "pie"}],

           [{}, {"type": 'pie'}]],

    subplot_titles=('Points by Endorser Category', 

                    'Points by Endorser Position', '% of Points by Endorser Party', 

                    'Number of Votes by Endorser Source', '% of Votes by Endorser State')

)





for i,e in enumerate(endorsee_df['endorsee'].unique()):

        

    visible = [False]*l*n_plots

    

    visible[i*lc:(i+1)*lc] = [True]*lc

        

    fig.add_trace(

            go.Bar(

                x=cat_df.loc[cat_df['endorsee']==e, 'category'],

                y=cat_df.loc[cat_df['endorsee']==e, 'pt_by_category'],

                text=cat_df.loc[cat_df['endorsee']==e, 'pt_by_category'],

                textposition='outside',

                opacity=0.9,

                marker={'color':colors[0],

                       'opacity':0.9},

                visible=False if i!=1 else True,

                showlegend=False),

        row=1, col=1)





    

    fig.add_trace(

            go.Bar(

                x=position_df.loc[position_df['endorsee']==e, 'position'],

                y=position_df.loc[position_df['endorsee']==e,'pt_by_position'],

                text=position_df.loc[position_df['endorsee']==e,'pt_by_position'],

                textposition='outside',

                opacity=0.9,

                marker={'color':colors[1],

                       'opacity':0.9},

                visible=False if i!=1 else True,

                showlegend=False),

        row=2, col=1)

    

    fig.add_trace(

            go.Pie(

                values=party_df.loc[party_df['endorsee']==e, 'pt_by_party'].to_numpy(),

                labels=party_df.loc[party_df['endorsee']==e, 'party'].to_numpy(),

                hole=0.4,

                visible=False if i!=1 else True,

                text=party_df.loc[party_df['endorsee']==e, 'party'],

                hoverinfo='label+percent+name',

                textinfo= 'percent+label',

                textposition = 'inside',

                showlegend=False,

                marker = dict(colors = plotly.colors.diverging.Geyser)),

        row=2, col=2)

    

    fig.add_trace(

            go.Bar(

                x=source_df.loc[source_df['endorsee']==e, 'source'],

                y=source_df.loc[source_df['endorsee']==e,'votes_by_source'],

                text=source_df.loc[source_df['endorsee']==e,'votes_by_source'],

                textposition='outside',

                opacity=0.9,

                marker={'color':colors[2],

                       'opacity':0.9},

                visible=False if i!=1 else True,

                showlegend=False

                       ),

        row=3, col=1)

    

    fig.add_trace(

            go.Pie(

                values=state_df.loc[state_df['endorsee']==e, 'votes_by_state'].to_numpy(),

                labels=state_df.loc[state_df['endorsee']==e, 'state'].to_numpy(),

                hole=0.4,

                visible=False if i!=1 else True,

                text=state_df.loc[state_df['endorsee']==e, 'full_state'],

                hoverinfo='label+percent+name',

                textinfo= 'percent+label',

                textposition = 'inside',

                showlegend=False,

                marker = dict(colors = plotly.colors.diverging.Geyser)),

        row=3, col=2)

    



    buttons.append(

        dict(label=e,

             method='update',

             args=[{'visible': visible},

                   #{'title': e}

                  ]))

    



fig.update_layout(

    title={'text': '<b> Endorsee Summary <b>', 'font':{'size':22},

            'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},

    margin=dict(t=150),

    height=1350,

    xaxis1=dict(tickangle=45, tickvals=cat_df['category'].unique(), ticktext=cat_df['category'].unique()),

    yaxis1=dict(range=[0, np.max(cat_df['pt_by_category']+15)]),

    

    xaxis2=dict(tickangle=45, tickvals=position_df['position'].unique(), ticktext=position_df['position'].unique()),

    yaxis2=dict(range=[0, np.max(position_df['pt_by_position']+15)]),

    

    xaxis3=dict(tickangle=45, tickvals=source_df['source'].unique(), ticktext=source_df['source'].unique()), 

    yaxis3=dict(range=[0, np.max(source_df['votes_by_source']+15)]), 

    

    bargap=0.1,

    showlegend=True,

    updatemenus = list([

        dict(active=1,

             buttons=buttons,

             direction="down",

             pad={"r": 10, "t": 10},

             showactive=True,

             x=-0.15,

             xanchor="left",

             y=1.04,

             yanchor="top"

         )

     ]))



fig['layout']['annotations'] += go.layout.Annotation(text="Select Endorsee", 

                                                     x=-0.15, xref="paper", 

                                                     y=1.05, yref="paper",

                                                     align="left", showarrow=False),

    

    



fig.show()
endorsee_df['date'] = pd.to_datetime(endorsee_df['date'])

e = endorsee_df.set_index('date')

pt_over_time = e.groupby("endorsee").resample('15D').agg({"endorser": np.size, "points": np.sum})

pt_over_time.reset_index(inplace=True)

pt_over_time['cum_points'] = pt_over_time.sort_values('date').groupby(by=['endorsee'])['points'].transform(lambda x: x.cumsum())

pt_over_time['cum_votes'] = pt_over_time.sort_values('date').groupby(by=['endorsee'])['endorser'].transform(lambda x: x.cumsum())
fig = go.Figure()



for i,e in enumerate(endorsee_df['endorsee'].unique()):

    

    fig.add_trace(

        go.Scatter(

            x=pt_over_time.loc[pt_over_time['endorsee']==e, 'date'],

            y=pt_over_time.loc[pt_over_time['endorsee']==e, 'cum_points'],

            name=e,

            mode ='markers+lines',

            showlegend=True)

        )

    

fig.update_layout(

    height=550,

    #width=800,

    title={'text': 'Total Points per over Time',

           'y':0.95, 'x':0.5,

           'xanchor': 'center', 'yanchor': 'top'},

    xaxis=dict(range=[date(2019,1,1), np.max(pt_over_time['date'])]),

    yaxis=dict(title='Points')

    )



fig.show()
cols = ['category', 'party', 'state']

d={}



for c in cols:

    tmp = endorsee_df.groupby(['endorsee', c]).agg({'points':'sum', 'endorser':'count'}).reset_index()

    tmp.rename(columns={'points': f'pt_by_{c}', 'endorser': f'votes_by_{c}'}, inplace=True)

    d[c] = tmp

    

    

n_plots=2

l=len(cols)

buttons=[]



fig = make_subplots(

    rows=2, cols=1,

    specs=[[{}],

           [{}]],

    row_heights=[0.65, 0.35]

)





for i,c in enumerate(cols):



    visible = [False]*l*n_plots

    visible[i*n_plots:(i+1)*n_plots] = [True]*n_plots



    tmp = d[c]

    

    fig.add_trace( 

        go.Scatter(

            y=tmp[c],

            x=tmp['endorsee'],

            mode='markers+text',

            marker=dict(

                size=np.where(tmp[f'pt_by_{c}']<50, tmp[f'pt_by_{c}']+20, 60),

                color=tmp[f'votes_by_{c}'],

                colorscale='geyser',

                showscale=False,

                opacity = 0.7),

            text=tmp[f'pt_by_{c}'],

            visible=True if i==0 else False,

            textposition='middle center'),

        row=1, col=1)



    fig.add_trace(

        go.Bar(

            x=tmp['endorsee'],

            y=tmp[f'pt_by_{c}'],

            text=tmp[f'pt_by_{c}'],

            hoverinfo='all',

            textposition='inside',

            visible=True if i==0 else False,

            marker=dict(

                color=tmp[f'pt_by_{c}'],

                colorscale='geyser')),

        row=2, col=1)



    buttons.append(

        dict(label= ' '.join([s.capitalize() for s in c.split('_')]),

             method='update',

             args=[{'visible': visible},

                   {'title': {'text': 'Points by ' + [s.capitalize() for s in c.split("_")][-1],

                              'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},

                   #'yaxis1': {'title': f'Endorser {c.capitalize()}'},

                   #'yaxis2': {'title': f'Points by Endorser {c.capitalize()}'}

                   }]

            )

        )



    

fig.update_layout(

      height=1350,

      showlegend=False,

      xaxis1=dict(tickangle=45, title='Endorsee'),

      #yaxis1=dict(title='Endorsee Category'),

      xaxis2=dict(tickangle=45, title='Endorsee'),

      yaxis2=dict(title='Points', type='log'),

      updatemenus = list([

          dict(active=0,

             buttons=buttons,

             direction="right",

             pad={"r": 10, "t": 10},

             #showactive=True,

             x=0.15,

             xanchor="left",

             y=1.08,

             yanchor="top"

         )

     ]),

    annotations=[

        go.layout.Annotation(text="Select Aggregation", x=-0.12, xref="paper", y=1.06, yref="paper",

                             align="left", showarrow=False),

    ])

    



fig.show()
noend = df.loc[df['endorsee']=='no_endorsee']

noend.head(10)
from itertools import product 

cols = ['category', 'party', 'position', 'state']

col_pairs=[]

d={}

for i,c1 in enumerate(cols):

    for c2 in cols[i+1:]:

        col_pair=c1.capitalize() + '-' + c2.capitalize()

        tmp = noend.groupby([c1,c2]).agg({'points':'sum', 'endorser':'count'}).reset_index()

        tmp.rename(columns={'points': f'pt_by_{col_pair}', 'endorser': f'votes_by_{col_pair}'}, inplace=True)

        d[col_pair] = tmp

        col_pairs.append((c1,c2))

l=len(col_pairs)

buttons=[]



fig = go.Figure()



for j, (c1,c2) in enumerate(col_pairs):

    

    col_pair = c1.capitalize() + '-' + c2.capitalize()

    visible = [False]*l

    visible[j] = True



    tmp = d[col_pair]

    

    fig.add_trace( 

        go.Scatter(

            x=tmp[c1],

            y=tmp[c2],

            mode='markers+text',

            marker=dict(

                size=np.where(tmp[f'pt_by_{col_pair}']<50, tmp[f'pt_by_{col_pair}']+20, 60),

                color=tmp[f'votes_by_{col_pair}'],

                colorscale='geyser',

                showscale=False,

                opacity = 0.7),

            text=tmp[f'pt_by_{col_pair}'],

            visible=True if j==0 else False,

            textposition='middle center'))



    buttons.append(

        dict(label=col_pair,

             method='update',

             args=[{'visible': visible},

                   {'title': {'text': f'Votes by <b>{c1.capitalize()}<b> &  <b>{c2.capitalize()}<b>',

                              'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},

                   'xaxis': {'title': c1.capitalize(), 'tickangle': 45},

                   'yaxis': {'title': c2.capitalize()}

                   }]

            )

        )

  



    

fig.update_layout(

      margin=dict(l=120, t=200),

      height=1150,

      showlegend=False,

      title = {'text': f'Votes by <b>{col_pairs[0][0].capitalize()}<b> &  <b>{col_pairs[0][1].capitalize()}<b>',

                              'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},

      xaxis={'title': col_pairs[0][0].capitalize(), 'tickangle':45},

      yaxis={'title': col_pairs[0][1].capitalize()},

      updatemenus = [

          go.layout.Updatemenu(

             active=0,

             buttons=buttons,

             direction="down",

             pad={"r": 10, "t": 10},

             #showactive=True,

             x=0,

             xanchor="left",

             y=1.1,

             yanchor="top"

         )

     ],

    annotations=[

        go.layout.Annotation(text="Select Columns", x=0.02, xref="paper", y=1.12, yref="paper",

                             align="left", showarrow=False),

    ])

    



fig.show()