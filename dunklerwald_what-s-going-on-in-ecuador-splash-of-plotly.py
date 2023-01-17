import numpy as np

import pandas as pd

import gc



import matplotlib.pyplot as plt

from IPython.core.display import HTML



import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots



import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 300)

pd.set_option("display.max_rows", 20)
df = pd.read_csv('../input/daily-temperature-of-major-cities/city_temperature.csv')

print(df.shape)

df.head()
#Сommon functions for exploratory data analysis

def get_stats(df):

    """

    Function returns a dataframe with the following stats for each column of df dataframe:

    - Unique_values

    - Percentage of missing values

    - Percentage of zero values

    - Percentage of values in the biggest category

    - data type

    """

    stats = []

    for col in df.columns:

        if df[col].dtype not in ['object', 'str', 'datetime64[ns]']:

            zero_cnt = df[df[col] == 0][col].count() * 100 / df.shape[0]

        else:

            zero_cnt = 0



        stats.append((col, df[col].nunique(),

                      df[col].isnull().sum() * 100 / df.shape[0],

                      zero_cnt,

                      df[col].value_counts(normalize=True, dropna=False).values[0] * 100,

                      df[col].dtype))



    df_stats = pd.DataFrame(stats, columns=['Feature', 'Unique_values',

                                            'Percentage of missing values',

                                            'Percentage of zero values',

                                            'Percentage of values in the biggest category',

                                            'type'])



    del stats

    gc.collect()



    return df_stats
get_stats(df)
del df['State']
print(f"Year           : min: {df['Year'].min()}, max {df['Year'].max()}")

print(f"Month          : min: {df['Month'].min()}, max {df['Month'].max()}")

print(f"Day            : min: {df['Day'].min()}, max {df['Day'].max()}")

print(f"AvgTemperature : min: {df['AvgTemperature'].min()}, max {df['AvgTemperature'].max()}")
df['Month'] = df['Month'].astype('int8')

df['Day'] = df['Day'].astype('int8')

df['Year'] = df['Year'].astype('int16')

df['AvgTemperature'] = df['AvgTemperature'].astype('float16')
print(f"There are {df[df['Day']==0].Day.count()} rows with Day=0")

df[df['Day']==0].head()
df['Year'].value_counts().sort_index()
df = df[df['Day']!=0]

df = df[~df['Year'].isin([200,201,2020])]

df = df.drop_duplicates()
df[df['Country']=='Equador'].groupby('Year')['AvgTemperature'].agg(['size','min','max','mean'])
df['AvgTemperature'].value_counts(normalize=True).head(5)
dfr = df[df['AvgTemperature']==-99]['Region'].reset_index().drop('index', axis=1)
( px.histogram(y=dfr['Region'])

 .update_layout(title_text='Distribution of AvgTemperature =-99.0 across regions', title_x=0.5)

 .update_xaxes(title_text='Row count')

 .update_yaxes(title_text=None)

 .update_traces(hovertemplate='<b>%{y}</b><br>count=%{x}<extra></extra>',opacity=0.75)

).show()
df = df[df['AvgTemperature']!=-99]
df['days_in_year']=df.groupby(['Country','Year'])['Day'].transform('size')

df[df['days_in_year']<=270]
df=df[df['days_in_year']>270]
df['Date'] = pd.to_datetime(df[['Year','Month', 'Day']])

df['AvgTemperature'] = (df['AvgTemperature'] -32)*(5/9)
code_dict = {'Czech Republic':'Czechia','Equador':'Ecuador', 'Ivory Coast':"Côte d'Ivoire",'Myanmar (Burma)':'Myanmar','Serbia-Montenegro':'Serbia', 'The Netherlands':'Netherlands'}

df['Country'].replace(code_dict, inplace=True)
print(f"Final data set shape: {df.shape}")
# global yearly stats:

# - average, min, max temperature per year 

# - date and location (city/country/region) of lowest temperature during this year

# - date and location (city/country/region) of highest temperature during this year

dfg = (

       df.groupby('Year')['AvgTemperature'].agg(['mean','min','idxmin','max','idxmax']).reset_index()

      .merge(df[['Region','Country','City','Date']], left_on='idxmin',right_index=True)

      .merge(df[['Region','Country','City','Date']], left_on='idxmax',right_index=True,suffixes=('_min','_max'))

      )



# top hottest/coldest cities over the entire period

dft = df.groupby(['Country','City'])['AvgTemperature'].mean().sort_values(ascending=False).reset_index()
fig = make_subplots(

     rows=2

    ,cols=2

    ,column_widths=[0.5, 0.5]

    ,row_heights=[0.5, 0.5]

    ,vertical_spacing=0.15

    ,specs=[[{"type": "scatter", "colspan": 2},None],

           [  {"type": "bar"}, {"type": "bar"}]]

    ,subplot_titles=['Global temperature trend (1995-2019)','Top 5 hottest cities','Top 5 coldest cities']

    ,y_title='Average temperature °C'

)



# global temperature trend graph

trace = (

          px.scatter(dfg, x='Year', y='mean',trendline='ols',trendline_color_override='red')

         .add_trace(px.line(dfg, x='Year', y='mean').data[0]) 

         .update_traces(hovertemplate='<b>%{x}</b><br><i>Avg temp :<b> %{y}</b></i><br>%{text}'

                        ,text = ['Min temp : <b>'+str(d['min'])+'</b>, country : '+d['Country_min']+', city : '+d['City_min']+', date : '+str(d['Date_min'])[:10] +'<br>'+'Max temp : <b>'+str(d['max'])+'</b>, country : '+d['Country_max']+', city : '+d['City_max']+', date :'+str(d['Date_max'])[:10]

                                 for _, d in dfg.iterrows()]

                        ,hoverlabel_bgcolor='white')

        ).data

fig.add_trace(trace[0], row=1, col=1)

fig.add_trace(trace[1], row=1, col=1)

fig.add_trace(trace[2], row=1, col=1)



# hottest cities graph

fig.add_trace(

    (

     px.bar(

             dft.head(5)

            ,x='City'

            ,y='AvgTemperature'

            ,color='AvgTemperature'

            ,color_continuous_scale=['darkorange','red']

            ,hover_data=['Country', 'AvgTemperature'] 

            ,opacity=0.8)

           ).data[0],

    row=2, col=1

)



# coldest cities graph

fig.add_trace(

   (

     px.bar(

             dft.tail(5)

            ,x='City'

            ,y='AvgTemperature'

            ,color='AvgTemperature'

            ,color_continuous_scale=['blue','lightblue']

            ,hover_data=['Country', 'AvgTemperature']

            ,title='Top 5 coldest cities'

            ,opacity=0.8)

           ).data[0],

    row=2, col=2

)



fig.update_layout(height=600, margin=dict(r=10, t=40, b=50, l=60))

fig.update_layout(coloraxis_autocolorscale=False, coloraxis_colorscale=['blue','lightblue','yellow','orange','darkorange','red'],coloraxis_colorbar_title='Temp °C')
iso_code = pd.read_csv('../input/iso-codes/iso_codes.csv')

iso_code = iso_code[['Country','ISO_Code']].drop_duplicates().reset_index(drop=True)

iso_code.head()
# temperature stats, grouped by country and year

dfc = (

       df.groupby(['Year','Country'])['AvgTemperature'].agg(['mean'])

      .reset_index()

      .rename(columns={'mean': 'AvgTemperature'})

      .merge(iso_code,left_on='Country',right_on='Country')

      .sort_values(by=['Year','Country'])

      )

dfc['Rank_hottest'] = dfc.groupby(by=['Year'])['AvgTemperature'].rank(method="min",ascending=False)

dfc['Rank_coldest'] = dfc.groupby(by=['Year'])['AvgTemperature'].rank(method="min",ascending=True)

dfc.head()
fig = (

   px.choropleth(

                 dfc               

                ,locations='ISO_Code'               

                ,color='AvgTemperature'

                ,hover_name='Country'  

                ,hover_data={'ISO_Code':False, 'Year':True,'AvgTemperature':':.2f'}

                ,animation_frame='Year'   

                ,color_continuous_scale='Portland' 

                ,height=600)

  .update_layout(

                 title_text='World average temperature dynamics'

                ,title_x=0.3

                ,margin=dict(r=10, t=40, b=10, l=10)

                ,coloraxis_colorbar_title='Temp °C')

)

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 800

fig.show()
# animation speed

step_duration=800



fig = make_subplots(

     rows=2

    ,cols=2

    ,shared_xaxes=False

    ,shared_yaxes=False

    ,column_widths=[0.5,0.5]

    ,row_heights=[0.2, 0.8]

    ,horizontal_spacing=0.05

    ,vertical_spacing=0.1 

    ,specs=[[{"type": "table"},{"type": "table"}], 

            [{"type": "bar"},{"type": "bar"}]]

    ,subplot_titles=[None,None,'Top coldest countries','Top hottest countries']

    ,y_title='Average temperature °C'

)



dfg_t=dfg[dfg['Year']==1995]



# graph for the lowest temperature day

fig.add_trace(

    go.Table(

        header=dict(

             values=list(['<b>' + 'Lowest temperature' + '</b>','',''])

            ,align="left"

            ,line_color='white'

            ,fill_color='white'

        ),

        cells=dict(

             values=['<b>' + dfg_t['min'].map(u"{:,.2f}".format) + '</b>', dfg_t['Date_min'].map(u"{:%Y-%m-%d}".format), dfg_t['City_min']+', '+dfg_t['Country_min']]

            ,align = "left"

            ,line_color='white'

            ,fill_color='white'

        )

    ),

    row=1, col=1

)



# graph for the highest temperature day

fig.add_trace(

    go.Table(

        header=dict(

             values=list(['<b>' + 'Highest temperature' + '</b>','',''])

            ,align="left"

            ,line_color='white'

            ,fill_color='white'

        ),

        cells=dict(

             values=['<b>' + dfg_t['max'].map(u"{:,.2f}".format) + '</b>', dfg_t['Date_max'].map(u"{:%Y-%m-%d}".format), dfg_t['City_max']+', '+dfg_t['Country_max']]

            ,align = "left"

            ,line_color='white'

            ,fill_color='white'

        )

    ),

    row=1, col=2

)



# top coldest countries graph

fig.add_trace(

    (

       px.bar(

              data_frame=dfc[dfc['Rank_coldest']<=5].sort_values(['Year','Rank_coldest'])

             ,x='Country'

             ,y='AvgTemperature'

             ,color='AvgTemperature'

             ,text='AvgTemperature'

             ,animation_frame='Year'

             ,opacity=0.8)

      .update_layout(

                     coloraxis_colorbar_title='Temp °C'

                    ,title_text='Top coldest countries'

                    ,title_x=0.5)

      .update_xaxes(title_text=None)

      .update_yaxes(title_text='Average temperature °C', range=[-4,33])               

      .update_traces(texttemplate='%{text:.2f}')

).data[0],

row=2, col=1

)



# top hottest countries graph

fig.add_trace(

    (

       px.bar(

              data_frame=dfc[dfc['Rank_hottest']<=5].sort_values(['Year','Rank_hottest'])

             ,x='Country'

             ,y='AvgTemperature'

             ,color='AvgTemperature'

             ,text='AvgTemperature'

             ,animation_frame='Year'

             ,opacity=0.8)

      .update_layout(

                     coloraxis_colorbar_title='Temp °C'

                    ,title_text='Top hottest countries'

                    ,title_x=0.5)

      .update_xaxes(title_text=None)

      .update_yaxes(title_text='Average temperature °C', range=[-4,33])               

      .update_traces(texttemplate='%{text:.2f}')

).data[0],

row=2, col=2

)



# animation frames

years = list(dfc['Year'].sort_values().unique())

frames=[]

for year in years: 

    dfg_t=dfg[dfg['Year']==year]

    dfc_c=dfc[(dfc['Rank_coldest']<=5)&(dfc['Year']==year)].sort_values(['Year','Rank_coldest'])

    dfc_h=dfc[(dfc['Rank_hottest']<=5)&(dfc['Year']==year)].sort_values(['Year','Rank_hottest'])

    

    frames.append(go.Frame(

                  name=str(year),

                  data=[

                        go.Table(cells=dict(

                            values=['<b>' + dfg_t['min'].map(u"{:,.2f}".format) + '</b>', dfg_t['Date_min'].map(u"{:%Y-%m-%d}".format), dfg_t['City_min']+', '+dfg_t['Country_min']]))

                       ,go.Table(cells=dict(

                            values=['<b>' + dfg_t['max'].map(u"{:,.2f}".format) + '</b>', dfg_t['Date_max'].map(u"{:%Y-%m-%d}".format), dfg_t['City_max']+', '+dfg_t['Country_max']]))

                       ,go.Bar(x=dfc_c['Country'], y=dfc_c['AvgTemperature'], text=dfc_c['AvgTemperature'])

                       ,go.Bar(x=dfc_h['Country'], y=dfc_h['AvgTemperature'], text=dfc_c['AvgTemperature'])

                      ],

                  traces=[0,1,2,3]))



fig.frames=frames



# buttons Play and Pause

buttons = [dict(

                 label='Play'

                ,method='animate'

                ,args=[  [f'{year}' for year in years[1:]]

                        ,dict(frame=dict(duration=step_duration, easing='linear', redraw=True)   

                        ,fromcurrent=True

                        ,transition=dict(duration=0, easing='linear'))])         

          ,dict(

                 label='Pause'

                ,method='animate'

                ,args=[  [None]

                        ,dict(frame=dict(duration=0, redraw=False)

                        ,mode='immediate'      

                        ,transition=dict(duration=0))])

          ]

# let's add buttons to the layout

updatemenus=[dict(

                   type='buttons'

                  ,direction='left'  

                  #,showactive=True 

                  ,y=0

                  ,x=-0.1

                  ,xanchor='left'

                  ,yanchor='top'

                  ,pad=dict(b=10, t=45) 

                  ,buttons=buttons)]



# yearly slider

sliders= [dict(

                yanchor='top'

               ,xanchor='left' 

               ,currentvalue=dict(prefix='Year: ', visible=True, xanchor='left')

               ,transition=dict(duration=0, easing='linear')

               ,pad=dict(b=10, t=25) 

               ,len=0.9, x=0.1, y=0 

               ,steps=[

                       dict(

                            args=[

                                   [year]

                                  ,dict(frame=dict(duration=step_duration, easing='linear', redraw=True)

                                  ,transition=dict(duration=0, easing='linear'))] 

                          ,label= str(year), method='animate')

                      for year in years       

                    ])]



fig.update_layout(updatemenus=updatemenus, sliders=sliders)

fig.update_layout(height=600,margin=dict(r=10, t=30, b=50, l=10))

fig.update_layout(coloraxis_autocolorscale=False, coloraxis_colorscale=['blue','lightblue','yellow','orange','darkorange','red'],coloraxis_colorbar_title='Temp °C')

fig.update_yaxes(range=[-4, 33], autorange=False, row=2, col=1)

fig.update_yaxes(range=[-4, 33], autorange=False, row=2, col=2)  
# temperature stats, grouped by region and year 

dfr = (

       df.groupby(['Year','Region'])['AvgTemperature'].agg(['mean','min','idxmin','max','idxmax']).reset_index()

      .merge(df[['Country','City','Date']], left_on='idxmin',right_index=True)

      .merge(df[['Country','City','Date']], left_on='idxmax',right_index=True,suffixes=('_min','_max'))

      )



# average temperature, smoothed with exponential weighted average.

dfr['mean_smoothed'] = dfr.groupby(['Region'])['mean'].transform(lambda x: x.ewm(span=3).mean()).fillna(dfr['mean'])
fig = make_subplots(

     rows=1

    ,cols=2

    ,column_widths=[0.5, 0.5]

    ,horizontal_spacing=0.05

    ,shared_yaxes=True

    ,specs=[[  {"type": "scatter"}, {"type": "scatter"}]]

    ,subplot_titles=['Original','Smoothed']

    ,y_title='Average temperature °C'

)



# temperature growth across different regions

traces = (

            px.line(dfr, x='Year', y='mean',color='Region', line_dash='Region')   

           .update_yaxes(title_text='Average temperature °C')

           .for_each_trace(

                 lambda trace: trace.update(hovertemplate='<b>%{x}</b><br><i>Avg temp :<b> %{y}</b></i><br>%{text}'

                               ,text = ['Min temp : <b>'+str(d['min'])+'</b>, country : '+d['Country_min']+', city : '+d['City_min']+', date : '+str(d['Date_min'])[:10] +'<br>'+'Max temp : <b>'+str(d['max'])+'</b>, country : '+d['Country_max']+', city : '+d['City_max']+', date :'+str(d['Date_max'])[:10]

                                   for _, d in dfr[dfr['Region']==trace.name].iterrows()]

                               ,hoverlabel_bgcolor='white'))

        ).data



for trace in traces:

    fig.add_trace(trace, row=1, col=1)



# temperature growth across different regions - smoothed version

traces = (

            px.line(dfr, x='Year', y='mean_smoothed',color='Region', line_dash='Region')   

           .update_yaxes(title_text='Average temperature °C')

           .for_each_trace(

                 lambda trace: trace.update(hovertemplate='<b>%{x}</b><br><i>Avg temp :<b> %{y}</b></i><br>%{text}'

                               ,text = ['Min temp : <b>'+str(d['min'])+'</b>, country : '+d['Country_min']+', city : '+d['City_min']+', date : '+str(d['Date_min'])[:10] +'<br>'+'Max temp : <b>'+str(d['max'])+'</b>, country : '+d['Country_max']+', city : '+d['City_max']+', date :'+str(d['Date_max'])[:10]

                                   for _, d in dfr[dfr['Region']==trace.name].iterrows()]

                               ,hoverlabel_bgcolor='white'))

        ).data



for trace in traces:

    trace.update(name = trace.name+' (smooth)')

    fig.add_trace(trace, row=1, col=2)

    

fig.update_layout(height=450, margin=dict(r=10, t=60, b=50, l=10), title_text="Temperature trend per region", title_x=0.22)

#fig.show()
# Temperature rise per region through the entire period, using exponentially smoothed average temperature

dfrs = dfr.groupby('Region')['mean_smoothed'].agg(['first','last']).reset_index()

dfrs['Temp_delta'] = dfrs['last'] - dfrs['first']

dfrs.columns=['Region','Start year temp','End year temp', 'Delta temp']



(

       px.bar(

              dfrs.sort_values(by='Region', ascending=False)

             ,y='Region'

             ,x='Delta temp'

             ,color='Delta temp'

             ,color_continuous_scale=['orange','red']

             ,text='Delta temp'

             ,hover_name='Region'

             ,hover_data={'Region':False,

                          'Delta temp':':.2f',

                          'Start year temp':':.2f', 

                          'End year temp':':.2f'})

      .update_layout(

                     coloraxis_colorbar_title='°C'

                    ,title_text='Temperature rise per region for the entire period of 1995-2019'

                    ,title_x=0.5)

      .update_xaxes(title_text='Temperature rise °C')  

      .update_yaxes(title_text='')  

      .update_traces(texttemplate='%{text:.2f}')

)
# Temperature rise per country through the entire period, using exponentially smoothed average temperature

dfc['AvgTemperature_smoothed'] = dfc.groupby(['Country'])['AvgTemperature'].transform(lambda x: x.ewm(span=3).mean()).fillna(dfc['AvgTemperature'])

dfcs = dfc.groupby('Country')['AvgTemperature_smoothed'].agg(['first','last']).reset_index()



dfcs['Temp_delta'] = dfcs['last'] - dfcs['first']

dfcs.columns=['Country','Start year temp','End year temp', 'Delta temp']

dfcs.head()
dfcsg = pd.concat([dfcs.sort_values(by='Delta temp', ascending=True).head(5), dfcs.sort_values(by='Delta temp', ascending=False).head(5)])

(

       px.bar(

              dfcsg.sort_values(by='Delta temp', ascending=False)

             ,y='Country'

             ,x='Delta temp'

             ,color='Delta temp'

             ,color_continuous_scale=['darkblue','blue','lightblue','orange','darkorange','red']

             ,text='Delta temp'

             ,hover_name='Country'

             ,hover_data={'Country':False,

                          'Delta temp':':.2f',

                          'Start year temp':':.2f', 

                          'End year temp':':.2f'}

             )

      .update_layout(

                     coloraxis_colorbar_title='Delta °C'

                    ,title_text='Top countries with the lowest and highest temperature rise for the entire period of 1995-2019'

                    ,title_x=0.5

                    ,height=400)

      .update_xaxes(title_text='Temperature delta °C')  

      .update_yaxes(title_text='')  

      .update_traces(texttemplate='%{text:.2f}')

)
df[df['Country']=='Ecuador'].groupby(['Year','City'])['AvgTemperature'].agg(['size','min','max','mean'])
# several mappings for seasonality charts

month_dict = {1:"January", 2:"February", 3:"March", 4:"April", 5:"May", 6:"June" ,7:"July", 8:"August", 9:"September", 10:"October", 11:"November", 12:"December"}

season_dict = {1:"Winter", 2:"Spring", 3:"Summer", 4:"Autumn"}

season_month_map = {1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4, 12:1}
# temperature stats, grouped by year, month, region and country 

dfmc = (

       df.groupby(['Year','Month','Region','Country'])['AvgTemperature'].agg(['mean'])

      .reset_index()

      .rename(columns={'mean': 'AvgTemperature','Month': 'Month_num'})

      .sort_values(by=['Year','Month_num','Region','Country'])

      )



dfmc['Season_num'] = dfmc['Month_num'].map(season_month_map)

dfmc['Season'] = dfmc['Season_num'].map(season_dict)

dfmc['Month'] = dfmc['Month_num'].map(month_dict)



# temperature stats, grouped by year, season, month and region 

dfmr = (

       dfmc.groupby(['Year','Season_num','Season','Month_num','Month','Region'])['AvgTemperature'].agg(['mean'])

      .reset_index()

      .rename(columns={'mean': 'AvgTemperature'})

      .sort_values(by=['Year','Month_num','Region'])

      )
(

    px.box(

            dfmc

           ,x='Region'

           ,y='AvgTemperature'

           ,color='Region')

   .update_layout(

                   title_text='Temperature distribution per region'

                  ,title_x=0.25

                  ,xaxis=dict(title_text=None, showticklabels=False)

                  ,yaxis=dict(title_text='Average temperature °C'))

)
# temperature stats, grouped by month and region 

dfmr_g = (

       dfmr.groupby(['Region','Month_num','Month'])['AvgTemperature'].agg(['mean'])

      .reset_index()

      .rename(columns={'mean': 'AvgTemperature'})

      .sort_values(by=['Region','Month_num'])

      )
(

   px.bar(

           dfmr_g

          ,x='Month'     

          ,y='AvgTemperature'

          ,facet_col='Region'

          ,facet_col_wrap=4

          ,facet_row_spacing=0.1

          ,color='Region'

          ,hover_name='Region'

          ,hover_data={'Region':False,'AvgTemperature':':.2f'}

          ,height=450

          ,width=800)

  .update_traces(showlegend=False)

  .update_layout(

                 title_text='Average temperature per region and month, °C'

                ,title_x=0.25          

                ,margin=dict(l=0,r=5))

  .update_xaxes(tickangle=-45) 

  .update_yaxes(title_text=None)  

  .for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

)    
# temperature stats, grouped by year, season and region 

dfsr = (

       dfmc.groupby(['Year','Season_num','Season','Region'])['AvgTemperature'].agg(['mean'])

      .reset_index()

      .rename(columns={'mean': 'AvgTemperature'})

      .sort_values(by=['Year','Season_num','Region'])

      )
(

   px.line(

           dfsr

          ,x='Year'     

          ,y='AvgTemperature'

          ,color='Region'

          ,facet_row='Season'

          ,facet_col='Region'

          ,facet_row_spacing=0.03

          ,hover_name='Region'

          ,hover_data={'Region':False,'Season':True,'AvgTemperature':':.2f'}

          ,height=450

          ,width=800)

  .update_traces(showlegend=False)

  .update_layout(

                 title_text='Seasonal temperature dynamics per region, °C'

                ,title_x=0.25

                ,margin=dict(r=40, t=60, b=50, l=0))

  .update_yaxes(title_text=None)  

  .for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1],textangle=0))

).show()    
dfmc.head()
dfyc = dfmc.groupby(['Country','Year'])['AvgTemperature'].mean().reset_index()

dfycs = dfmc.groupby(['Country','Year','Season_num','Season'])['AvgTemperature'].mean().reset_index()



# add new "period" dimension: 1995-2014 (first 15 years) and 2015-2019 (last 5 years) 

dfmc['Period'] = '1995-2014'

dfmc['Period'].loc[dfmc['Year']>2014] = '2015-2019'



fig = make_subplots(

     rows=4

    ,cols=4

    ,row_heights=[0.25, 0.25, 0.15, 0.35]

    ,vertical_spacing=0.1

    ,horizontal_spacing=0.02

    ,shared_yaxes=True

    ,specs=[[{"type": "scatter", "colspan": 4},None,None,None]

            ,[{"type": "scatter"},{"type": "scatter"},{"type": "scatter"},{"type": "scatter"}]

            ,[{"type": "histogram", "colspan": 4},None,None,None]

            ,[{"type": "histogram", "colspan": 4},None,None,None]]

    ,subplot_titles=[

                      'Average temperature dynamics on country level (1995-2019)'

                     ,'winter','spring','summer','autumn'

                     ,'Temperature distribution dynamics: (1995-2014) vs (2015-2019)'

                     ,None]

)



# average temperature dynamics on country level (1995-2019) subplot

# initially all subplots output information for the first country in the list of countries (=dfyc['Country'].head(1))

fig.add_trace(

    (

       px.line(

           data_frame = dfyc[dfyc['Country']==dfyc['Country'].head(1).squeeze()]

          ,x='Year'

          ,y='AvgTemperature')  

).data[0],

row=1, col=1

)





# seasonal dynamics subplots

traces_seasonal = (

       px.line(

           data_frame = dfycs[dfycs['Country']==dfyc['Country'].head(1).squeeze()]

          ,x='Year'     

          ,y='AvgTemperature'

          ,facet_col='Season'

          ,color='Season'

          ,color_discrete_sequence=['blue','green','red','orange'])

      .update_traces(showlegend=False)

      .for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1],textangle=0)) 

).data

for i, trace in enumerate(traces_seasonal):

    fig.add_trace(trace, row=2, col=i+1)



    

# temperature distribution dynamics: (1995-2014) vs (2015-2019) subplot   

traces_dist = (

           px.histogram(

                         dfmc[dfmc['Country']==dfyc['Country'].head(1).squeeze()]

                        ,x='AvgTemperature'

                        ,histnorm='probability density'

                        ,color='Period'

                        ,barmode='overlay'

                        ,marginal='box'

                        ,nbins=25)

          .update_traces(showlegend=False)

          .for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1],textangle=0)) 

).data

fig.add_trace(traces_dist[0], row=4, col=1)

fig.add_trace(traces_dist[1], row=3, col=1)

fig.add_trace(traces_dist[2], row=4, col=1)

fig.add_trace(traces_dist[3], row=3, col=1)





buttons = []

# populate frames in all subplots for each country in the dropdown list

for country in dfyc['Country'].sort_values().unique():

    # average temperature dynamics on country level subplot

    dfyc_c = dfyc[dfyc['Country']==country]

    args_x=[dfyc_c['Year']]

    args_y=[dfyc_c['AvgTemperature']]

    args_f=[0]

    # seasonal subplots

    for i in range(len(traces_seasonal)):

        dfycs_c = dfycs[(dfycs['Country']==country)&(dfycs['Season_num']==i+1)]

        args_x.append(dfycs_c['Year'])

        args_y.append(dfycs_c['AvgTemperature'])

        args_f.append(i+1)

    # temperature distribution dynamics subplot

    period_num = len(dfmc['Period'].sort_values().unique())

    frames_num = round(len(traces_dist)/period_num)

    k = 0

    for j, period in enumerate(dfmc['Period'].sort_values().unique()):

        for i in range(frames_num):   

            dfmc_c = dfmc[(dfmc['Country']==country)&(dfmc['Period']==period)]

            args_x.append(dfmc_c['AvgTemperature'])

            args_y.append(None)

            args_f.append(len(traces_seasonal)+k+1) 

            k += 1

    

    buttons.append(dict(method='restyle',

                        label=country,

                        visible=True,

                        args=[{'x': args_x,'y':args_y}, args_f]

                        )

                  )    





# update layout menu with our country drill down box and country frames

updatemenu=[dict(

                   buttons=buttons

                  ,direction='down'

                  ,pad={'r': 10, 't': 10}

                  ,showactive=True

                  ,x=-0.05

                  ,xanchor='left'

                  ,y=1.1

                  ,yanchor='top')] 

                              



fig.data[0].line.dash='dash'

fig.data[0].mode ='markers+lines' 

fig.data[0].line.color='#00CC96' 

fig.update_layout(font_size=10)

fig.for_each_annotation(lambda a: a.update(font=dict(size=14)))

fig.layout.annotations[0].font.size=16

fig.layout.annotations[-1].font.size=16

fig.update_xaxes(range=[1995, 2019], autorange=False, row=1)

fig.update_xaxes(range=[1995, 2019], autorange=False, row=2) 

fig.update_xaxes(showticklabels=False, row=3)

fig.update_yaxes(showticklabels=False, row=3)



fig.update_traces(showlegend=True, selector=dict(type='histogram'))

fig.update_layout( updatemenus=updatemenu

                  ,height=600

                  ,barmode='overlay'

                  ,margin=dict(r=10, t=20, b=30, l=0)

                  ,legend=dict(

                               orientation='h'

                              ,yanchor='top'

                              ,y=0.33

                              ,xanchor='left'

                              ,x=-0.05

                            ))



del dfyc_c, dfycs_c, dfmc_c, dfyc, dfycs

gc.collect()



fig.show()